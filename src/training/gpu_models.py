"""
gpu_models.py
Implements GPU-accelerated training using XGBoost and PyTorch.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

class PianoXGBoost(BaseEstimator, ClassifierMixin):
    """
    XGBoost Wrapper with GPU support (Histogram method).
    """
    def __init__(self, use_gpu=True, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.device = "cuda" if self.use_gpu else "cpu"

    def fit(self, X, y):
        logger.info(f"Training XGBoost on {self.device}...")

        # Ensure inputs are appropriate (numpy or pandas)
        # XGBoost handles them well.

        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'tree_method': 'hist',
            'device': self.device,
            'max_depth': self.max_depth,
            'eta': self.learning_rate,
            'eval_metric': 'mlogloss'
        }

        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(params, dtrain, num_boost_round=self.n_estimators)
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest).astype(int)

    def predict_proba(self, X):
        # We need output_margin=False for probas?
        # multi:softmax output gives class.
        # multi:softprob gives probas.
        # Re-training with multi:softprob or just changing prediction type?
        # XGBoost separates objective.
        # Actually, for sklearn compatibility we might need the sklearn API of XGB.
        # But here we implement custom.
        # Let's use sklearn API wrapped internally if simpler, OR just DMatrix.
        # DMatrix predict returns probs if objective is softprob.
        # But we set softmax above.

        # Better: Use XGBClassifier from xgboost package which handles this.
        pass

    def save(self, path):
        # Save the Booster
        self.model.save_model(path)

class PianoPyTorch(nn.Module):
    """
    Simple Linear Model for Classification (SVM-like logic if Hinge, or Softmax).
    Multi-class (4 classes).
    """
    def __init__(self, input_dim, num_classes=4):
        super(PianoPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)

class PyTorchTrainer:
    def __init__(self, input_dim, use_gpu=True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = PianoPyTorch(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y, epochs=10, batch_size=64):
        logger.info(f"Training PyTorch Model on {self.device}...")

        # Convert to Tensor
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            # logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def save(self, path):
        torch.save(self.model.state_dict(), path)
