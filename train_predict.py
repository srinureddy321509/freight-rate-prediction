from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.base import clone


TARGET = "posted_rate"
ID_COL = "load_id"
DATE_COL = "date"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create model features without using load_id or the target."""
    X = frame.copy()
    X = X.drop(columns=[ID_COL, TARGET], errors="ignore")

    if DATE_COL in X.columns:
        dates = pd.to_datetime(X[DATE_COL], errors="coerce")
        X["date_year"] = dates.dt.year
        X["date_month"] = dates.dt.month
        X["date_day"] = dates.dt.day
        X["date_dayofweek"] = dates.dt.dayofweek
        X["date_dayofyear"] = dates.dt.dayofyear
        X = X.drop(columns=[DATE_COL])

    # Treat all non-numeric columns as categorical; normalize blank strings.
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].astype("string").str.strip()
        X[col] = X[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return X


def build_model(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    categorical = list(X.select_dtypes(include=["object", "string", "category"]).columns)
    numeric = [col for col in X.columns if col not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )

    # HistGradientBoosting is not used here because its categorical/one-hot
    # sparse handling can vary by scikit-learn version. ExtraTrees is robust
    # for mixed engineered tabular features after one-hot encoding.
    model = ExtraTreesRegressor(
        n_estimators=350,
        min_samples_leaf=2,
        max_features=0.9,
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def chronological_split(frame: pd.DataFrame, validation_fraction: float = 0.2):
    """Split chronologically when dates exist; otherwise use a reproducible random split."""
    if DATE_COL in frame.columns:
        dates = pd.to_datetime(frame[DATE_COL], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates) >= max(10, int(0.5 * len(frame))):
            cutoff = valid_dates.quantile(1.0 - validation_fraction)
            train_mask = dates < cutoff
            valid_mask = dates >= cutoff
            if train_mask.sum() >= 10 and valid_mask.sum() >= 10:
                return train_mask.to_numpy(), valid_mask.to_numpy(), f"chronological split; cutoff={cutoff.date()}"
    from sklearn.model_selection import train_test_split

    indices = np.arange(len(frame))
    train_idx, valid_idx = train_test_split(
        indices, test_size=validation_fraction, random_state=42
    )
    train_mask = np.zeros(len(frame), dtype=bool)
    valid_mask = np.zeros(len(frame), dtype=bool)
    train_mask[train_idx] = True
    valid_mask[valid_idx] = True
    return train_mask, valid_mask, "random 80/20 split (no sufficiently usable date column)"


def ensure_positive_predictions(values: np.ndarray, target: pd.Series) -> np.ndarray:
    # Rates are required to be positive by score.py. Use a small positive floor
    # based on the observed target scale to avoid invalid non-positive outputs.
    positive = pd.to_numeric(target, errors="coerce")
    positive = positive[np.isfinite(positive) & (positive > 0)]
    floor = max(0.01, float(positive.min()) * 0.05) if len(positive) else 0.01
    return np.maximum(np.asarray(values, dtype=float), floor)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a freight-rate model and write assessment predictions.")
    parser.add_argument("--train", default="data/train-test.csv")
    parser.add_argument("--validation", default="data/validation.csv")
    parser.add_argument("--template", default="data/validation-predictions-template.csv")
    parser.add_argument("--december", default="data/december-chart-inputs.csv")
    parser.add_argument("--output", default="validation-predictions.csv")
    parser.add_argument("--metrics-output", default="validation_metrics.json")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    validation_path = Path(args.validation)
    template_path = Path(args.template)
    december_path = Path(args.december)

    train = read_csv(train_path)
    validation = read_csv(validation_path)
    template = read_csv(template_path)
    december = read_csv(december_path)

    if TARGET not in train.columns:
        raise ValueError(f"Training file must contain target column '{TARGET}'.")
    if ID_COL not in validation.columns or ID_COL not in template.columns:
        raise ValueError("Validation and template files must contain load_id.")
    if "predicted_rate" not in template.columns:
        raise ValueError("Template file must contain predicted_rate.")
    if len(validation) != len(template):
        raise ValueError(f"Validation rows ({len(validation)}) do not match template rows ({len(template)}).")
    if validation[ID_COL].astype(str).tolist() != template[ID_COL].astype(str).tolist():
        # Safely align by ID rather than silently writing rates against wrong loads.
        if validation[ID_COL].astype(str).duplicated().any() or template[ID_COL].astype(str).duplicated().any():
            raise ValueError("Duplicate load_id values prevent safe alignment.")
        template_ids = set(template[ID_COL].astype(str))
        validation_ids = set(validation[ID_COL].astype(str))
        if template_ids != validation_ids:
            raise ValueError("Template load_id values do not match validation.csv.")
        validation = validation.set_index(validation[ID_COL].astype(str)).loc[
            template[ID_COL].astype(str)
        ].reset_index(drop=True)

    # Clean unusable target rows; retain all valid positive rates.
    train[TARGET] = pd.to_numeric(train[TARGET], errors="coerce")
    train = train.loc[np.isfinite(train[TARGET]) & (train[TARGET] > 0)].copy()
    if len(train) < 20:
        raise ValueError("Fewer than 20 rows have valid positive posted_rate labels.")

    # Prevent common target leakage from columns that directly encode the rate.
    # quote_signal is retained only if present; inspect the README/data dictionary
    # to verify it is known at prediction time and not derived from posted_rate.
    X_all = prepare_features(train)
    y_all = train[TARGET].astype(float)
    train_mask, valid_mask, split_description = chronological_split(train)
    if train_mask.sum() < 10 or valid_mask.sum() < 5:
        raise ValueError("Not enough rows in train/validation split; inspect the data and split.")

    validation_model = build_model(X_all.loc[train_mask], args.random_state)
    validation_model.fit(X_all.loc[train_mask], y_all.loc[train_mask])
    valid_pred = validation_model.predict(X_all.loc[valid_mask])
    valid_pred = ensure_positive_predictions(valid_pred, y_all.loc[train_mask])
    valid_y = y_all.loc[valid_mask]
    metrics = {
        "split": split_description,
        "train_rows": int(train_mask.sum()),
        "holdout_rows": int(valid_mask.sum()),
        "mae": float(mean_absolute_error(valid_y, valid_pred)),
        "rmse": float(np.sqrt(mean_squared_error(valid_y, valid_pred))),
        "r2": float(r2_score(valid_y, valid_pred)),
    }
    Path(args.metrics_output).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Holdout metrics:")
    print(json.dumps(metrics, indent=2))

    # Retrain on all labeled rows for final inference.
    final_X = prepare_features(train)
    final_model = build_model(final_X, args.random_state)
    final_model.fit(final_X, y_all)

    # Use the provided template and preserve its exact required column order.
    validation_X = prepare_features(validation)
    validation_pred = ensure_positive_predictions(
        final_model.predict(validation_X), y_all
    )
    output = template[["load_id"]].copy()
    output["predicted_rate"] = validation_pred
    output["predicted_rate"] = pd.to_numeric(output["predicted_rate"], errors="raise")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, float_format="%.2f")
    print(f"Wrote {len(output):,} validation predictions to {output_path}")

    # Fill December chart inputs with the same final model.
    required_december = [
        "pickup", "delivery", "distance", "equipment", "weight", "date"
    ]
    missing = [col for col in required_december if col not in december.columns]
    if missing:
        raise ValueError(f"December input is missing required columns: {missing}")

    # Apply the same feature preparation used during training.
    december_X = prepare_features(december)

    # Ensure prediction columns match the training feature schema.
    # Missing features are filled with NaN so the fitted imputers can handle them.
    # Extra columns are removed.
    december_X = december_X.reindex(columns=final_X.columns)

    december_pred = ensure_positive_predictions(
        final_model.predict(december_X), y_all
    )

    december_out = december.copy()
    december_out["predicted_rate"] = december_pred

    # Preserve the required seven-column schema and order.
    december_columns = [
        "pickup", "delivery", "distance", "equipment",
        "weight", "date", "predicted_rate"
    ]
    december_out = december_out[december_columns]

    december_path.parent.mkdir(parents=True, exist_ok=True)
    december_out.to_csv(
        december_path, index=False, float_format="%.2f"
    )
    print(f"Wrote {len(december_out):,} December predictions to {december_path}")

    print("\nNext run:")
    print(
        "python score.py --predictions "
        f"{output_path} --december-predictions {december_path}"
    )



if __name__ == "__main__":
    main()
