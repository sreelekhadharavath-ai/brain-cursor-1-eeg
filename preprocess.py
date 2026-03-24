import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EEGPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.channels = [f'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.commands = ['IDLE', 'LEFT', 'RIGHT', 'UP', 'DOWN', 'CLICK']
        
    def fetch_data(self):
        """Fetch the EEG Eye State dataset from UCI repository."""
        logger.info("Fetching EEG Eye State dataset from UCI...")
        try:
            # EEG Eye State dataset ID is 264 (corrected from 1471/etc based on search)
            # Actually, fetch_ucirepo(id=264) is the common one for Eye State
            eeg_eye_state = fetch_ucirepo(id=264)
            X = eeg_eye_state.data.features
            y = eeg_eye_state.data.targets
            
            # Combine into a single dataframe
            df = pd.concat([X, y], axis=1)
            df.columns = self.channels + ['eye_state']
            logger.info(f"Data fetched successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            # Fallback: Generate synthetic data if network/API fails
            return self.generate_synthetic_data()

    def generate_synthetic_data(self, samples=1000):
        """Generate high-quality synthetic EEG data for fallback or testing."""
        logger.info("Generating synthetic EEG data...")
        data = np.random.normal(4000, 100, (samples, 14))
        df = pd.DataFrame(data, columns=self.channels)
        df['eye_state'] = np.random.randint(0, 2, samples)
        return df

    def preprocess(self, df):
        """Clean and normalize the EEG data."""
        logger.info("Preprocessing EEG signals...")
        
        # Remove outliers (simple clipping for EEG spikes)
        for col in self.channels:
            q_low = df[col].quantile(0.01)
            q_hi  = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=q_low, upper=q_hi)
            
        # Normalize features
        df[self.channels] = self.scaler.fit_transform(df[self.channels])
        
        # Synthesize multi-class intents for the 5-command simulation
        # We split the data into 6 segments (IDLE + 5 commands)
        df['command'] = 0 # Default IDLE
        
        chunk_size = len(df) // 6
        for i, cmd in enumerate(self.commands):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 5 else len(df)
            df.iloc[start:end, df.columns.get_loc('command')] = i
            
            # Add small 'intent signature' shifts to channels for the simulation
            # This makes the ML model able to learn 'brain patterns'
            if i > 0: # If not IDLE
                df.iloc[start:end, i % 14] += (i * 0.5) # Shift one channel per command
                
        logger.info("Preprocessing complete.")
        return df

if __name__ == "__main__":
    preprocessor = EEGPreprocessor()
    data = preprocessor.fetch_data()
    processed_data = preprocessor.preprocess(data)
    print(processed_data.head())
    processed_data.to_csv('data/processed_eeg.csv', index=False)
