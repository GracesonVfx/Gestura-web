import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import glob
import os
import json

print("TensorFlow Version:", tf.__version__)

def main():
    csv_files = glob.glob('hand_landmarks_dataset*.csv')

    if not csv_files:
        print("Error: Could not find any hand_landmarks_dataset CSVs.")
        print("Please use the Web App to record your dual-hand dataset first!")
        return

    dfs = []
    for file in csv_files:
        try:
            df_temp = pd.read_csv(file)
            if 'label' in df_temp.columns:
                dfs.append(df_temp)
                print(f"Loaded {len(df_temp)} samples from {file}")
        except Exception as e:
            print(f"Skipping {file} due to error: {e}")

    if not dfs:
        print("No valid data found in the CSVs.")
        return

    # Merge all CSVs
    df = pd.concat(dfs, ignore_index=True)

    if len(df) == 0:
        print("Dataset is empty. Run collect data first.")
        return

    # Prepare Features (X) and Labels (y)
    X = df.drop('label', axis=1).values
    y_raw = df['label'].values

    # Encode labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    print(f"\nDataset loaded. Total samples: {len(X)}")
    print(f"Features per sample: {X.shape[1]}")
    if X.shape[1] != 126:
        print(f"WARNING: Expected 126 features (63 left + 63 right), but got {X.shape[1]}. Make sure you recorded using the updated Web App!")

    print(f"Classes found ({num_classes}): {list(encoder.classes_)}")

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the Neural Network Model
    # Since these are static frames, a Dense Neural Network works well.
    print("\nBuilding and Training Keras Model...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.2), # Prevent overfitting
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        # Output layer with softmax for probability distribution over classes
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Train the model
    # Early stopping stops training if the validation loss doesn't improve for 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%")

    # Export to TFLite
    print("\nExporting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Saved 'model.tflite'")

    # Export a simple JSON Weights file for the Web UI
    # Since tfjs pip package is broken on python 3.13, we manually export the dense weights.
    print("\nExporting Keras weights to JSON format...")
    tfjs_target_dir = '../hand-tracker-web/public/tfjs_model'
    if not os.path.exists(tfjs_target_dir):
        os.makedirs(tfjs_target_dir)
        
    custom_model_data = {
        "classes": list(encoder.classes_),
        "layers": []
    }
    
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights, biases = layer.get_weights()
            custom_model_data["layers"].append({
                "type": "Dense",
                "units": layer.units,
                "activation": layer.activation.__name__,
                "weights": weights.tolist(),
                "biases": biases.tolist()
            })
            
    with open(os.path.join(tfjs_target_dir, 'weights.json'), 'w') as f:
        json.dump(custom_model_data, f)
    print(f"Saved JSON weights to {tfjs_target_dir}/weights.json")

    # Save Labels
    with open('labels.txt', 'w') as f:
        for cls in encoder.classes_:
            f.write(f"{cls}\n")
    print("Saved 'labels.txt'")

    print("\nSuccess! You can now use model.tflite in your Flutter app, and test the tfjs_model in your Web App.")

if __name__ == "__main__":
    main()
