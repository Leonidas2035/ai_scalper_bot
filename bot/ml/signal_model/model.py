import xgboost as xgb
import pickle


class SignalModel:

    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="mlogloss"
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, features: dict):
        import numpy as np

        arr = np.array([list(features.values())])
        pred = self.model.predict(arr)[0]
        prob = self.model.predict_proba(arr)[0]

        return pred, prob

    def save(self, path="signal_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path="signal_model.pkl"):
        with open(path, "rb") as f:
            self.model = pickle.load(f)
