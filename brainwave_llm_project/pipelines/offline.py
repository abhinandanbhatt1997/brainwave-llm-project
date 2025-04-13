from utils.preprocess import load_raw_eeg, preprocess_eeg, extract_features
from utils.train import train_model
import numpy as np

def run_offline_pipeline():
    # Load dataset
    raw = load_raw_eeg("data/raw/eegmmidb/S001/S001R01.edf")
    epochs = preprocess_eeg(raw)
    features = extract_features(epochs)
    
    # Synthetic labels (replace with real labels)
    labels = np.random.randint(0, 2, len(features))
    
    # Train and evaluate
    accuracy = train_model(features, labels)
    print(f"Model accuracy: {accuracy:.2f}")