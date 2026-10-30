import os
import sys
import pickle
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

# Ensure current and parent directories are in sys.path for clean imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_prep import prepare_data, load_prepared_data, SteamFeatureTransformer


def resolve_split_path(data_dir: str, split_name: str) -> str:
    """Resolves path to cleaned split CSV file across directory levels."""
    candidates = [
        os.path.join(data_dir, split_name, "games.csv"),
        os.path.join(parent_dir, data_dir, split_name, "games.csv"),
        os.path.join(current_dir, "..", data_dir, split_name, "games.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return os.path.abspath(candidates[0])


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    early_stopping_rounds: int = 15,
    n_estimators: int = 300
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads already-prepared train, valid, and test sets directly from cleaned_data/
    (prepared previously by data_prep.py), trains XGBoost with validation early stopping
    and native categorical support (enable_categorical=True), evaluates on the holdout test set,
    and persists the model bundle.
    """
    train_path = resolve_split_path(data_dir, "train")
    valid_path = resolve_split_path(data_dir, "valid")
    test_path = resolve_split_path(data_dir, "test")

    save_dir_abs = os.path.abspath(model_save_dir if os.path.isabs(model_save_dir) else os.path.join(parent_dir, model_save_dir))
    transformer_path = os.path.join(save_dir_abs, "feature_transformer.pkl")

    # Check if cleaned data splits and feature transformer exist; regenerate if missing
    need_regenerate = False
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path) and os.path.exists(transformer_path)):
        need_regenerate = True

    if need_regenerate:
        print(f"Regenerating cleaned & prepared splits via data_prep.py...")
        prepare_data(output_dir=data_dir, model_save_dir=save_dir_abs)

    print(f"Loading already-prepared splits from: '{os.path.dirname(os.path.dirname(train_path))}'...")
    (train_df, valid_df, test_df), transformer = load_prepared_data(
        data_dir=data_dir,
        transformer_path=transformer_path
    )

    print(f"Train split size: {len(train_df):,} rows")
    print(f"Valid split size: {len(valid_df):,} rows")
    print(f"Test split size:  {len(test_df):,} rows")

    # Directly extract pre-computed feature matrices and target vectors
    X_train = train_df[transformer.feature_names]
    y_train = train_df["is_hit"].values

    X_val = valid_df[transformer.feature_names]
    y_val = valid_df["is_hit"].values

    X_test = test_df[transformer.feature_names]
    y_test = test_df["is_hit"].values

    cat_cols_in_matrix = [c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)]
    print(f"Native Categorical Columns ({len(cat_cols_in_matrix)}): {cat_cols_in_matrix}")
    print(f"Total Feature Matrix Columns: {X_train.shape[1]}")
    print(f"Feature matrix shape (Train): {X_train.shape}")

    # Class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

    # Train XGBoost Classifier with Native Categorical Support & Early Stopping
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds
    )

    print(f"Training XGBoost with native categorical support (early_stopping_rounds={early_stopping_rounds})...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"Early stopping best iteration: {model.best_iteration} (Best validation logloss: {model.best_score:.4f})")

    # Holdout test set evaluation
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else n_estimators,
        "best_val_logloss": float(model.best_score) if hasattr(model, "best_score") else None
    }

    print("\n================ Holdout Test Set Evaluation ================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Flop (0)", "Hit (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================\n")

    # Feature importances summary
    importances = pd.Series(model.feature_importances_, index=transformer.feature_names).sort_values(ascending=False)
    print("Top 15 Most Important Features:")
    for rank, (feat, val) in enumerate(importances.head(15).items(), 1):
        print(f"  {rank:2d}. {feat:<30} {val:.4f}")

    # Persist model artifact
    os.makedirs(save_dir_abs, exist_ok=True)
    model_artifact_path = os.path.join(save_dir_abs, "xgboost_hit_predictor.pkl")

    artifact = {
        "model": model,
        "transformer": transformer,
        "transformer_state": transformer.to_dict(),
        "feature_names": transformer.feature_names,
        "cat_cols": transformer.cat_cols,
        "cat_categories": transformer.cat_categories,
        "metrics": metrics
    }

    with open(model_artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSuccessfully saved updated model artifact to: {model_artifact_path}")
    return model, metrics


if __name__ == "__main__":
    train_xgboost()
