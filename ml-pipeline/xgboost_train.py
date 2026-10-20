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
    for genres and platforms, alongside price and LLM vibe scores.
    """
    def __init__(self, encoder: OrdinalEncoder = None):
        self.encoder = encoder
        self.feature_names = [
            "price",
            "genres_encoded",
            "platforms_encoded",
            "hype_score",
            "story_score",
            "nostalgia_score"
        ]

    def _canonicalize_delimited(self, series: pd.Series) -> pd.Series:
        return series.astype(str).apply(
            lambda val: ";".join(sorted([token.strip().lower() for token in str(val).split(";") if token.strip()]))
        )

    def fit(self, df: pd.DataFrame):
        genres_clean = self._canonicalize_delimited(df["genres"])
        platforms_clean = self._canonicalize_delimited(df["platforms"])
        
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
        self.encoder.fit(pd.DataFrame({"genres": genres_clean, "platforms": platforms_clean}))
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_copy = df.copy()
        
        # If ordinal columns already precomputed in dataset
        if "genres_encoded" in df_copy.columns and "platforms_encoded" in df_copy.columns:
            genres_enc = df_copy["genres_encoded"].fillna(-1.0).values.reshape(-1, 1)
            platforms_enc = df_copy["platforms_encoded"].fillna(-1.0).values.reshape(-1, 1)
        else:
            genres_clean = self._canonicalize_delimited(df_copy["genres"])
            platforms_clean = self._canonicalize_delimited(df_copy["platforms"])
            encoded = self.encoder.transform(
                pd.DataFrame({"genres": genres_clean, "platforms": platforms_clean})
            )
            genres_enc = encoded[:, 0].reshape(-1, 1)
            platforms_enc = encoded[:, 1].reshape(-1, 1)

        price = df_copy["price"].fillna(0.0).astype(float).values.reshape(-1, 1)
        hype = df_copy["hype_score"].fillna(5.0).astype(float).values.reshape(-1, 1)
        story = df_copy["story_score"].fillna(5.0).astype(float).values.reshape(-1, 1)
        nostalgia = df_copy["nostalgia_score"].fillna(5.0).astype(float).values.reshape(-1, 1)

        features = np.hstack([price, genres_enc, platforms_enc, hype, story, nostalgia])
        return features


def train_xgboost(
    data_dir: str = "cleaned_data",
    model_save_dir: str = "saved_models/ml_models"
) -> Tuple[xgb.XGBClassifier, Dict[str, Any]]:
    """
    Loads train, valid, and test sets with Ordinal Encoded genres and platforms,
    trains the XGBoost Classifier with validation monitoring, evaluates on holdout test set,
    and persists the model bundle.
    """
    train_path = os.path.join(data_dir, "train.csv")
    valid_path = os.path.join(data_dir, "valid.csv")
    test_path = os.path.join(data_dir, "test.csv")
    encoder_path = os.path.join(model_save_dir, "ordinal_encoder.pkl")
    
    # If cleaned splits or ordinal columns are missing, regenerate pipeline
    if not (os.path.exists(train_path) and os.path.exists(valid_path) and os.path.exists(test_path)):
        print(f"Cleaned splits not found in '{data_dir}/'. Triggering full data preparation pipeline...")
        train_df, valid_df, test_df, encoder = prepare_data(output_dir=data_dir, encoder_save_path=encoder_path)
    else:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
        test_df = pd.read_csv(test_path)
        if "genres_encoded" not in train_df.columns:
            print("Regenerating cleaned splits with Ordinal Encoding...")
            train_df, valid_df, test_df, encoder = prepare_data(output_dir=data_dir, encoder_save_path=encoder_path)
        elif os.path.exists(encoder_path):
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
    
    print("Training XGBoost Classifier using Ordinal Encoded features...")
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
    
    print("\n================ Holdout Test Set Evaluation (Ordinal Encoding) ================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Flop (0)", "Hit (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("===============================================================================\n")
    
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
