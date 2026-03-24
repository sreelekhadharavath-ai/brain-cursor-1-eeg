import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGModelHandler:
    def __init__(self, model_path='data/bci_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.commands = ['IDLE', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'CLICK']

    def train(self, data_path='data/processed_eeg.csv'):
        """Train the ExtraTrees model on EEG data."""
        logger.info(f"Loading training data from {data_path}...")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file {data_path} not found. Run preprocess.py first.")
            
        df = pd.read_csv(data_path)
        X = df.drop(['eye_state', 'command'], axis=1)
        y = df['command']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info("Training ExtraTreesClassifier (optimized for size)...")
        self.model = ExtraTreesClassifier(n_estimators=50, random_state=42)
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        logger.info(f"Model Accuracy: {acc:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=self.commands))
        
        # Save model using joblib with compression
        joblib.dump(self.model, self.model_path, compress=3)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load the pre-trained model."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info("Model loaded successfully.")
        else:
            logger.warning("No pre-trained model found.")

    def predict(self, eeg_features):
        """Predict cursor command from 14-channel EEG input."""
        if self.model is None:
            self.load_model()
            
        if self.model is not None:
            # Ensure input is 2D
            if isinstance(eeg_features, pd.DataFrame):
                features = eeg_features
            else:
                features = np.array(eeg_features).reshape(1, -1)
                
            prediction = self.model.predict(features)[0]
            return self.commands[int(prediction)]
        return "IDLE"

if __name__ == "__main__":
    handler = EEGModelHandler()
    try:
        handler.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
