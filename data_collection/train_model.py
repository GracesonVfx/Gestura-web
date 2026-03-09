import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json

# File paths
json_model_output = '../hand-tracker-web/public/model.json'

print("Loading dataset(s)...")
import glob
csv_files = glob.glob('hand_landmarks_dataset*.csv')

if not csv_files:
    print("Error: Could not find any hand_landmarks_dataset CSVs. Please use the Web App to collect data and download the CSV here.")
    exit(1)

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        if 'label' in df.columns:
            dfs.append(df)
            print(f"Loaded {len(df)} samples from {file}")
    except Exception as e:
        print(f"Skipping {file} due to error: {e}")

if not dfs:
    print("No valid data found in the CSVs.")
    exit(1)

# Merge all CSVs
df = pd.concat(dfs, ignore_index=True)

# Ensure there's data
if len(df) == 0:
    print("Dataset is empty. Run collect_data.py first.")
    exit(1)

# Prepare Features (X) and Labels (y)
X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset loaded. Total samples: {len(df)}")
print(f"Classes found: {y.unique()}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Random Forest Classifier...")
# We use a simple but robust Random Forest.
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

# We want to use this model directly in our browser JS app.
# Since we only rely on the structure of the trees, we can export the trees to JSON.
print(f"\nExporting model to {json_model_output}...")

def rf_to_json(rf, feature_names, class_names):
    """
    Exports a scikit-learn RandomForestClassifier to a custom structured JSON format
    that can be easily parsed by JavaScript.
    """
    forest = []
    
    # Iterate over all DecisionTrees
    for idx, tree in enumerate(rf.estimators_):
        tree_ = tree.tree_
        
        def traverse(node):
            if tree_.children_left[node] == tree_.children_right[node]: # Leaf node
                # Get the class with the highest probability
                val = tree_.value[node][0]
                class_idx = np.argmax(val)
                return {"type": "leaf", "class": class_names[class_idx]}
                
            # Internal node: get feature index and threshold
            return {
                "type": "node",
                "feature": int(tree_.feature[node]),
                "threshold": float(tree_.threshold[node]),
                "left": traverse(tree_.children_left[node]),
                "right": traverse(tree_.children_right[node])
            }
            
        forest.append(traverse(0))
        
    return {
        "classes": list(class_names),
        "trees": forest
    }

# Export
custom_json = rf_to_json(model, X.columns.tolist(), model.classes_)

with open(json_model_output, 'w') as f:
    json.dump(custom_json, f)

print("Export successful! The Vite app can now load this model to recognize signs.")
