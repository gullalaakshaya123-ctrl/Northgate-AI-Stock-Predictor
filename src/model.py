from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from .market import FEATURES, make_features

def train_and_evaluate(history: pd.DataFrame, horizon: int = 5):
    """Chronological split to avoid randomly mixing future observations into training."""
    frame = make_features(history, horizon).dropna(subset=["target_return"])
    frame = frame.dropna(subset=["return_1d"])
    if len(frame) < 120:
        raise ValueError("Not enough history to train reliably. Choose a longer period or another ticker.")
    split = int(len(frame) * 0.8)
    train, test = frame.iloc[:split], frame.iloc[split:]
    models = {
        "Random Forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(n_estimators=250, max_depth=7, min_samples_leaf=4,
                                  random_state=42, n_jobs=-1)
        ),
        "Gradient Boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=42)
        )
    }
    results, fitted = [], {}
    X_train, X_test = train[FEATURES], test[FEATURES]
    y_train, y_test = train["target_return"], test["target_return"]
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results.append({
            "Model": name,
            "MAE (return)": mean_absolute_error(y_test, pred),
            "RMSE (return)": mean_squared_error(y_test, pred) ** 0.5,
            "R²": r2_score(y_test, pred)
        })
        fitted[name] = model
    best_name = min(results, key=lambda row: row["MAE (return)"])["Model"]
    best = fitted[best_name]
    latest_features = make_features(history, horizon).dropna(subset=FEATURES).iloc[[-1]][FEATURES]
    predicted_return = float(best.predict(latest_features)[0])
    last_price = float(history["close"].iloc[-1])
    return pd.DataFrame(results), best_name, predicted_return, last_price * (1 + predicted_return), len(train), len(test)

def get_feature_importance(model):
    estimator = model[-1]
    if hasattr(estimator, "feature_importances_"):
        return pd.DataFrame({"Feature": FEATURES, "Importance": estimator.feature_importances_}).sort_values("Importance", ascending=False)
    return pd.DataFrame({"Feature": FEATURES, "Importance": np.zeros(len(FEATURES))})
