import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb

from data_prep import prepare_data


class SteamFeatureTransformer:
    """
    Transforms game features into a structured DataFrame suitable for XGBoost training
    with native categorical support (enable_categorical=True).
    Covers all categorical columns from the dataset:
    developer, publisher, platforms, categories, genres, steamspy_tags, owners.
    """
    def __init__(self, cat_categories: Dict[str, List[Any]] = None):
        self.cat_cols = [
            "developer",
            "publisher",
            "platforms",
            "categories",
            "genres"
        ]
        self.num_cols = [
            "price",
            "english",
            "required_age",
            "achievements"
        ]
        self.feature_names = self.num_cols + self.cat_cols
        self.cat_categories = cat_categories or {}

    def _canonicalize_delimited(self, series: pd.Series) -> pd.Series:
        return series.astype(str).apply(
            lambda val: ";".join(sorted([token.strip().lower() for token in str(val).split(";") if token.strip()]))
        )

    def _clean_text_series(self, series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().fillna("Unknown")

    def fit(self, df: pd.DataFrame):
        self.cat_categories = {}
        for col in self.cat_cols:
            if col in df.columns:
                if col in ["genres", "categories", "steamspy_tags", "platforms"]:
                    s = self._canonicalize_delimited(df[col])
                else:
                    s = self._clean_text_series(df[col])
            else:
                s = pd.Series(["Unknown"] * len(df))
            cats = sorted(s.unique().tolist())
            self.cat_categories[col] = cats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        result = pd.DataFrame(index=df_copy.index)

        # Numeric columns
        for col in self.num_cols:
            if col in df_copy.columns:
                result[col] = pd.to_numeric(df_copy[col], errors="coerce").fillna(0.0)
            elif col == "price" and "price_scaled" in df_copy.columns:
                result[col] = pd.to_numeric(df_copy["price_scaled"], errors="coerce").fillna(0.0)
            else:
                result[col] = 0.0

        # Categorical columns cast to category dtype
        for col in self.cat_cols:
            if col in df_copy.columns:
                if col in ["genres", "categories", "steamspy_tags", "platforms"]:
                    s = self._canonicalize_delimited(df_copy[col])
                else:
                    s = self._clean_text_series(df_copy[col])
            else:
                s = pd.Series(["Unknown"] * len(df_copy), index=df_copy.index)

            cats = self.cat_categories.get(col, sorted(s.unique().tolist()))
            s_clean = s.where(s.isin(cats), other=None)
            result[col] = pd.Categorical(s_clean, categories=cats)

        return result


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models",
    early_stopping_rounds: int = 15
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads train, valid, and test sets, prepares all categorical columns for native XGBoost handling
    (enable_categorical=True), trains with validation early stopping, evaluates on holdout test set,
    and persists the model bundle.
    """
    train_path = os.path.join(data_dir, "train", "games.csv")
    valid_path = os.path.join(data_dir, "valid", "games.csv")
    test_path = os.path.join(data_dir, "test", "games.csv")
    transformer_path = os.path.join(model_save_dir, "feature_transformer.pkl")
    
    # Check if cleaned data splits exist
    need_regenerate = False
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        need_regenerate = True
            
    if need_regenerate:
        print(f"Regenerating cleaned splits...")
        attr_splits, desc_splits, scaler = prepare_data(output_dir=data_dir)
        train_df, valid_df, test_df = attr_splits
    else:
        print(f"Loading cleaned splits from '{data_dir}/'...")
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)

    print(f"Train split size: {len(train_df):,} rows")
    print(f"Valid split size: {len(valid_df):,} rows")
    print(f"Test split size:  {len(test_df):,} rows")
    
    y_train = train_df["is_hit"].values
    y_val = valid_df["is_hit"].values
    y_test = test_df["is_hit"].values
    
    # Initialize and fit transformer across all categorical & numerical columns
    transformer = SteamFeatureTransformer()
    transformer.fit(train_df)
        
    X_train = transformer.transform(train_df)
    X_val = transformer.transform(valid_df)
    X_test = transformer.transform(test_df)
    
    print(f"Categorical Columns ({len(transformer.cat_cols)}): {transformer.cat_cols}")
    print(f"Numerical Columns ({len(transformer.num_cols)}):   {transformer.num_cols}")
    print(f"Total Feature Columns: {transformer.feature_names}")
    print(f"Feature matrix shape (Train): {X_train.shape}")
    
    # Class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    # Train XGBoost Classifier with Native Categorical Support & Early Stopping
    model = xgb.XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        n_estimators=300,
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
        "f1_score": f1
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
    
    # Persist model artifact
    os.makedirs(model_save_dir, exist_ok=True)
    model_artifact_path = os.path.join(model_save_dir, "xgboost_hit_predictor.pkl")
    
    artifact = {
        "model": model,
        "transformer": transformer,
        "cat_categories": transformer.cat_categories,
        "feature_names": transformer.feature_names,
        "metrics": metrics
    }
    
    with open(model_artifact_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Successfully saved updated model artifact to: {model_artifact_path}")
    return model, metrics


if __name__ == "__main__":
    train_xgboost()
