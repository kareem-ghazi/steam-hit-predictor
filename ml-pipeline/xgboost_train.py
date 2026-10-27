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
    Transforms game features into a numerical feature matrix using Ordinal Encoding
    for genres, platforms, and publisher, alongside price, english flag, and LLM vibe scores.
    """
    def __init__(self, encoder: OrdinalEncoder = None):
        self.encoder = encoder
        self.feature_names = [
            "price",
            "english",
            "genres_encoded",
            "platforms_encoded",
            "publisher_encoded",
            "hype_score",
            "story_score",
            "nostalgia_score"
        ]

    def _canonicalize_delimited(self, series: pd.Series) -> pd.Series:
        return series.astype(str).apply(
            lambda val: ";".join(sorted([token.strip().lower() for token in str(val).split(";") if token.strip()]))
        )

    def _clean_text_series(self, series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().fillna("Unknown")

    def fit(self, df: pd.DataFrame):
        genres_clean = self._canonicalize_delimited(df["genres"])
        platforms_clean = self._canonicalize_delimited(df["platforms"])
        publisher_clean = self._clean_text_series(df["publisher"])
        
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.encoder.fit(pd.DataFrame({
            "genres": genres_clean,
            "platforms": platforms_clean,
            "publisher": publisher_clean
        }))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_copy = df.copy()
        
        # If ordinal columns already precomputed in dataset
        if all(col in df_copy.columns for col in ["genres_encoded", "platforms_encoded", "publisher_encoded"]):
            genres_enc = df_copy["genres_encoded"].fillna(-1.0).values.reshape(-1, 1)
            platforms_enc = df_copy["platforms_encoded"].fillna(-1.0).values.reshape(-1, 1)
            publisher_enc = df_copy["publisher_encoded"].fillna(-1.0).values.reshape(-1, 1)
        else:
            genres_clean = self._canonicalize_delimited(df_copy["genres"])
            platforms_clean = self._canonicalize_delimited(df_copy["platforms"])
            publisher_clean = self._clean_text_series(df_copy.get("publisher", pd.Series(["Unknown"] * len(df_copy))))
            encoded = self.encoder.transform(
                pd.DataFrame({
                    "genres": genres_clean,
                    "platforms": platforms_clean,
                    "publisher": publisher_clean
                })
            )
            genres_enc = encoded[:, 0].reshape(-1, 1)
            platforms_enc = encoded[:, 1].reshape(-1, 1)
            publisher_enc = encoded[:, 2].reshape(-1, 1)

        price = df_copy["price"].fillna(0.0).astype(float).values.reshape(-1, 1)
        english = df_copy["english"].fillna(1).astype(float).values.reshape(-1, 1) if "english" in df_copy.columns else np.ones((len(df_copy), 1), dtype=float)
        hype = df_copy["hype_score"].fillna(5.0).astype(float).values.reshape(-1, 1)
        story = df_copy["story_score"].fillna(5.0).astype(float).values.reshape(-1, 1)
        nostalgia = df_copy["nostalgia_score"].fillna(5.0).astype(float).values.reshape(-1, 1)

        features = np.hstack([price, english, genres_enc, platforms_enc, publisher_enc, hype, story, nostalgia])
        return features


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models"
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads train, valid, and test sets with Ordinal Encoded genres, platforms, and publisher,
    trains the XGBoost Classifier with validation monitoring, evaluates on holdout test set,
    and persists the model bundle.
    """
    train_path = os.path.join(data_dir, "train", "games.csv")
    valid_path = os.path.join(data_dir, "valid", "games.csv")
    test_path = os.path.join(data_dir, "test", "games.csv")
    encoder_path = os.path.join(model_save_dir, "ordinal_encoder.pkl")
    
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
        if os.path.exists(encoder_path):
            with open(encoder_path, "rb") as f:
                encoder = pickle.load(f)
        else:
            encoder = None

    print(f"Train split size: {len(train_df)} rows")
    print(f"Valid split size: {len(valid_df)} rows")
    print(f"Test split size:  {len(test_df)} rows")
    
    y_train = train_df["is_hit"].values
    y_val = valid_df["is_hit"].values
    y_test = test_df["is_hit"].values
    
    # Initialize and fit transformer
    transformer = SteamFeatureTransformer(encoder=encoder)
    if transformer.encoder is None:
        transformer.fit(train_df)
        
    X_train = transformer.transform(train_df)
    X_val = transformer.transform(valid_df)
    X_test = transformer.transform(test_df)
    
    print(f"Model Feature Columns: {transformer.feature_names}")
    print(f"Feature matrix shape (Train): {X_train.shape}")
    
    # Class imbalance weighting
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)
    
    # Train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )
    
    print("Training XGBoost Classifier using Ordinal Encoded features (genres, platforms, publisher) + english...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
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
        "encoder": transformer.encoder,
        "feature_names": transformer.feature_names,
        "metrics": metrics
    }
    
    with open(model_artifact_path, "wb") as f:
        pickle.dump(artifact, f)
        
    print(f"Successfully saved updated model artifact to: {model_artifact_path}")
    return model, metrics


if __name__ == "__main__":
    train_xgboost()
