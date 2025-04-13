from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

def train_model(features: np.ndarray, labels: np.ndarray):
    """Train and save classifier"""
    X_train, X_test, y_train, y_test = train_test_split(features, labels)
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    joblib.dump(clf, "models/classifier.pkl")
    return clf.score(X_test, y_test)