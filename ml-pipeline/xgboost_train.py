import os
import sys
import pickle
import argparse
import fnmatch
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from data_prep import prepare_data, load_prepared_data


def resolve_split_path(data_dir: str, split_name: str) -> str:
    """Resolves path to cleaned split CSV file across directory levels."""
    opts = [
        os.path.join(data_dir, split_name, "games.csv"),
        os.path.join(PARENT_DIR, data_dir, split_name, "games.csv"),
        os.path.join(CURRENT_DIR, "..", data_dir, split_name, "games.csv"),
    ]
    return next((os.path.abspath(o) for o in opts if os.path.exists(o)), os.path.abspath(opts[0]))


def filter_training_features(
    all_features: List[str],
    exempt_features: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """
    Filters features for training, supporting exact names and wildcard patterns (e.g. 'tag_*').
    Returns (selected_features, matched_exempt_features).
    """
    if not exempt_features:
        return list(all_features), []

    exempt_set = set()
    for pattern in exempt_features:
        p = pattern.strip()
        if not p:
            continue
        if any(c in p for c in ["*", "?", "["]):
            matched = [f for f in all_features if fnmatch.fnmatch(f, p)]
            exempt_set.update(matched)
        else:
            if p in all_features:
                exempt_set.add(p)

    selected = [f for f in all_features if f not in exempt_set]
    matched_exempt = [f for f in all_features if f in exempt_set]
    return selected, matched_exempt


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    early_stopping_rounds: int = 15,
    n_estimators: int = 300,
    exempt_features: Optional[List[str]] = None
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads pre-prepared splits directly from cleaned_data/, trains XGBoost with early stopping
    and native categorical support, evaluates on the holdout test set, and saves the model artifact.
    Allows controlling which features are exempt from training.
    """
    train_path = resolve_split_path(data_dir, "train")
    valid_path = resolve_split_path(data_dir, "valid")
    test_path = resolve_split_path(data_dir, "test")
    save_dir_abs = os.path.abspath(model_save_dir if os.path.isabs(model_save_dir) else os.path.join(PARENT_DIR, model_save_dir))
    transformer_path = os.path.join(save_dir_abs, "feature_transformer.pkl")

    # Ensure cleaned data exists
    if not all(os.path.exists(p) for p in [train_path, valid_path, test_path, transformer_path]):
        print("Pre-prepared splits or transformer missing; running data_prep.py...")
        prepare_data(output_dir=data_dir, model_save_dir=save_dir_abs)

    print(f"Loading prepared splits from: '{os.path.dirname(os.path.dirname(train_path))}'...")
    (train_df, valid_df, test_df), transformer = load_prepared_data(data_dir=data_dir, transformer_path=transformer_path)

    # Resolve training features with exemption control
    feature_cols, matched_exempt = filter_training_features(transformer.feature_names, exempt_features)
    if matched_exempt:
        print(f"Exempting {len(matched_exempt)} features from training: {matched_exempt}")
    print(f"Active Training Features ({len(feature_cols)} of {len(transformer.feature_names)}):")

    X_train = train_df[feature_cols]
    y_train = train_df["is_hit"].values
    X_val = valid_df[feature_cols]
    y_val = valid_df["is_hit"].values
    X_test = test_df[feature_cols]
    y_test = test_df["is_hit"].values

    cat_cols_in_matrix = [c for c in X_train.columns if isinstance(X_train[c].dtype, pd.CategoricalDtype)]
    print(f"Native Categorical Columns ({len(cat_cols_in_matrix)}): {cat_cols_in_matrix}")
    print(f"Training Matrix Shape: {X_train.shape}")

    # Class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

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

    print(f"Training XGBoost (early_stopping_rounds={early_stopping_rounds})...")
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    if hasattr(model, "best_iteration") and model.best_iteration is not None:
        print(f"Early stopping best iteration: {model.best_iteration} (Val logloss: {model.best_score:.4f})")

    # Holdout evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1,
        "best_iteration": model.best_iteration if hasattr(model, "best_iteration") else n_estimators,
        "best_val_logloss": float(model.best_score) if hasattr(model, "best_score") else None,
        "active_features_count": len(feature_cols),
        "exempt_features": matched_exempt
    }

    print("\n================ Holdout Test Set Evaluation ================")
    print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Flop (0)", "Hit (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("=============================================================\n")

    # Top features
    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("Top 15 Most Important Features:")
    for rank, (feat, val) in enumerate(importances.head(15).items(), 1):
        print(f"  {rank:2d}. {feat:<30} {val:.4f}")

    # Save artifact
    os.makedirs(save_dir_abs, exist_ok=True)
    model_artifact_path = os.path.join(save_dir_abs, "xgboost_hit_predictor.pkl")
    artifact = {
        "model": model,
        "transformer": transformer,
        "transformer_state": transformer.to_dict(),
        "training_features": feature_cols,
        "feature_names": feature_cols,
        "exempt_features": matched_exempt,
        "cat_cols": cat_cols_in_matrix,
        "cat_categories": transformer.cat_categories,
        "metrics": metrics
    }
    with open(model_artifact_path, "wb") as f:
        pickle.dump(artifact, f)

    print(f"\nSaved model artifact to: {model_artifact_path}")
    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Steam Hit Predictor")
    parser.add_argument("--data-dir", default="cleaned_data", help="Directory with prepared splits")
    parser.add_argument("--model-save-dir", default="saved_models/ml_models", help="Model artifact destination")
    parser.add_argument("--early-stopping-rounds", type=int, default=15)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument(
        "--exempt-features", nargs="*", default=None,
        help="List or space/comma-separated features to exclude from training (supports wildcards, e.g. 'tag_*')"
    )
    args = parser.parse_args()

    # Parse exempt features if provided
    exempt_list = []
    if args.exempt_features:
        for item in args.exempt_features:
            for sub in item.split(","):
                if sub.strip():
                    exempt_list.append(sub.strip())

    train_xgboost(
        data_dir=args.data_dir,
        model_save_dir=args.model_save_dir,
        early_stopping_rounds=args.early_stopping_rounds,
        n_estimators=args.n_estimators,
        exempt_features=exempt_list if exempt_list else None
    )
