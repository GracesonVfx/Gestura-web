# Gestura - AI Sign Language Recognition (Dual Hand)

This repository contains the full end-to-end pipeline for tracking **both hands** simultaneously, capturing their spatial coordinates, and training deep neural networks to translate the movements into Sign Language.

## Architecture Pipeline
This project is split into two modules:
1. **Frontend Data Collector (`hand-tracker-web`)**: A powerful React/Vite web application that uses MediaPipe to extract 126 skeletal landmarks across both the left and right hand from your webcam in real-time, exporting them to localized CSVs.
2. **Backend AI Trainer (`data_collection/train_tflite_model.py`)**: A Python neural network pipeline that ingests the dual-hand data, trains a Keras model, and exports it to both `.tflite` (for Flutter/Mobile deployment) and a raw `weights.json` (for native Web browser execution).

---

## 🚀 Step 1: Web App Setup & Data Collection

You use the web app to easily collect high-quality data natively in your browser.  
No Python MediaPipe installation is required!

### Requirements
- Node.js installed

### Installation & Run
```bash
cd hand-tracker-web
npm install
npm run dev
```

### Collecting Data
1. Open the web app `http://localhost:5173/`. 
2. Enter a letter (e.g. `A` or `B`) in the Data Collection Panel.
3. Click **Record CSV**. Hold your hands steady! It will capture exactly 126 coordinates (left and right) for 200 frames. 
4. Your browser will instantly download a `hand_landmarks_dataset (A).csv` file.
5. Move those CSV files to the `data_collection/` directory.

---

## 🧠 Step 2: Training the AI Model

Once you have your CSV datasets, it's time to train the machine learning model.

### Requirements
- Python 3.10+ (Tested on Python 3.13)

### Installation
Open a terminal in the `data_collection` directory and install the following data-science libraries:
```bash
cd data_collection
pip install pandas numpy scikit-learn
pip install tf-nightly
```
*(Note: We use tf-nightly because standard TensorFlow distributions are not fully compatible with Python 3.13 yet).*

### Train & Export
Run the custom builder script:
```bash
python train_tflite_model.py
```
**This script automatically handles the entire pipeline:**
1. Ingests all `hand_landmarks_dataset (*).csv` files in the folder.
2. Formats and scales the 126 coordinates.
3. Trains a deep sequential neural network.
4. Detects the maximum accuracy.
5. **Exports `model.tflite`** and `labels.txt` directly into the `data_collection/` folder for immediate use in Native Mobile/Flutter apps.
6. **Exports `tfjs_model/weights.json`** directly into the `hand-tracker-web/public/` folder so your Web App instantly updates its live predictions on localhost.

You can now refresh the Web App and it will automatically apply the newly generated intelligence!
