from brainflow.board_shim import BoardShim, BrainFlowInputParams
from utils.preprocess import extract_features
from utils.llm import query_llm
import joblib
import numpy as np

def run_realtime_pipeline():
    # Load model
    clf = joblib.load("models/classifier.pkl")
    
    # Setup Muse headset
    params = BrainFlowInputParams()
    params.serial_port = "COM4"  # Update in config/settings.py
    board = BoardShim(BoardIds.MUSE_2_BOARD, params)
    
    board.prepare_session()
    board.start_stream()
    
    try:
        while True:
            data = board.get_current_data(256)  # 1 sec of data
            features = extract_features(data)    # Requires adaptation for real-time
            state = clf.predict(features.reshape(1, -1))[0]
            print(query_llm(state))
    finally:
        board.stop_stream()
        board.release_session()