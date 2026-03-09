import json
import os
import glob
import time
import numpy as np

class StorageManager:
    def __init__(self, signs_dir="signs"):
        self.signs_dir = signs_dir
        if not os.path.exists(self.signs_dir):
            os.makedirs(self.signs_dir)

    def save_sign(self, letter, landmarks_sequence, metadata=None):
        """
        Saves a 30-frame sequence of landmarks for a given letter.
        landmarks_sequence shape: (30, 42, 3)
        """
        pattern = os.path.join(self.signs_dir, f"{letter}_v*.json")
        existing_versions = glob.glob(pattern)
        
        version = len(existing_versions) + 1
        filename = f"{letter}_v{version}.json"
        
        if len(existing_versions) == 0 and not os.path.exists(os.path.join(self.signs_dir, f"{letter}.json")):
            filename = f"{letter}.json"
            
        filepath = os.path.join(self.signs_dir, filename)
        
        data = {
            "letter": letter,
            "hand_used": metadata.get("which_hand_dominant", "unknown") if metadata else "unknown",
            "hand_symmetry": metadata.get("hand_symmetry", False) if metadata else False,
            "finger_states": metadata.get("finger_states", []) if metadata else [],
            "landmarks": landmarks_sequence.tolist(), # Convert numpy array to list
            "created_at": time.time(),
            "version": version
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
            
        return filepath

    def load_all_signs(self):
        """
        Loads all stored signs into memory.
        Returns a dictionary grouping sequences by letter.
        """
        all_signs = {}
        for filename in os.listdir(self.signs_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.signs_dir, filename)
                with open(filepath, 'r') as f:
                    try:
                        data = json.load(f)
                        letter = data["letter"]
                        
                        if letter not in all_signs:
                            all_signs[letter] = []
                            
                        # Convert list back to numpy array
                        data["landmarks"] = np.array(data["landmarks"])
                        all_signs[letter].append(data)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        
        return all_signs

    def list_signs(self):
        """Returns a summary of stored signs."""
        signs = self.load_all_signs()
        summary = {}
        for letter, items in signs.items():
            summary[letter] = len(items)
        return summary

    def delete_sign(self, letter):
        """Deletes all versions of a specific sign."""
        deleted_count = 0
        for filename in os.listdir(self.signs_dir):
            if filename.startswith(f"{letter}.json") or filename.startswith(f"{letter}_v"):
                filepath = os.path.join(self.signs_dir, filename)
                os.remove(filepath)
                deleted_count += 1
        return deleted_count
