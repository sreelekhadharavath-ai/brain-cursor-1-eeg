# Brain-Controlled Cursor Movement System 🧠🖱️
### Assistive Hands-Free Computing using EEG Signals

This project is a production-quality simulation of an end-to-end Brain-Computer Interface (BCI). It translates raw EEG (Electroencephalography) signals into computer cursor commands, enabling hands-free interaction for individuals with physical disabilities.

## 🌟 Key Features
- **Real-time EEG Sensing:** Simulates high-frequency signal acquisition from a 14-channel EEG headset.
- **Advanced Signal Processing:** Includes noise handling, normalization, and outlier removal using `scikit-learn`.
- **ExtraTrees Intent Detection:** A high-precision ensemble model trained to distinguish between:
  - `IDLE`, `LEFT`, `RIGHT`, `UP`, `DOWN`, and `CLICK`.
- **Assistive Cursor Control:** Direct integration with `pyautogui` for system-level cursor movement.
- **Professional Dashboard:** A premium Streamlit UI for monitoring cognitive states and signal streams.

## 🏗️ System Architecture
1. **EEG Device Layer:** Signal fetching from the UCI EEG Eye State dataset (ID: 264).
2. **Preprocessing Layer:** Outlier clipping and `StandardScaler` normalization.
3. **Inference Engine:** ExtraTrees Classifier predicting intent at 10Hz.
4. **Command Mapping:** Translation of intent into absolute/relative cursor offsets.
5. **Execution Layer:** `pyautogui` automation with safety fail-safes.
6. **Visualization:** Real-time line charts and cognitive metrics.

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.8+
- Windows OS (for `pyautogui` cursor control)

### 2. Installation
```bash
# Clone the repository (if applicable)
cd "brain eeg"

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage
First, ensure the model is trained:
```bash
python preprocess.py
python model.py
```

Then, launch the BCI Dashboard:
```bash
streamlit run app.py
```

## 📂 Project Structure & GitHub Upload List
The following files should be uploaded to your GitHub repository:

| File / Folder | Description |
| :--- | :--- |
| `app.py` | Main Streamlit UI and simulation loop. |
| `model.py` | Intent detection model training and prediction logic. |
| `preprocess.py` | Data acquisition and signal cleaning. |
| `controller.py` | Logic for mapping brain signals to cursor movements. |
| `requirements.txt` | Project dependencies (includes `joblib` for model compression). |
| `README.md` | Project documentation and instructions. |
| `.gitignore` | Config to exclude temporary files (e.g., `__pycache__`). |
| `data/bci_model.pkl` | The trained model (~4.5MB - optimized for size). |
| `data/processed_eeg.csv` | Sample processed EEG data (optional, for simulation). |

## ⚠️ Safety Warning
This system uses `pyautogui` to control your mouse. 
- **Failsafe:** Move the mouse cursor to any corner of the screen to abort movements.
- Ensure you have the terminal window backgrounded but accessible when testing.

---
*Developed for Assistive Technology Research & Development.*
