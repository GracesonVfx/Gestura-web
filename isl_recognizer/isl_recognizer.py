import cv2
import time
import numpy as np
import os
import argparse

from landmark_utils import LandmarkExtractor
from storage_manager import StorageManager
from dtw_matcher import DTWMatcher
from visualizer import Visualizer

class ISLRecognizerApp:
    def __init__(self):
        self.extractor = LandmarkExtractor()
        self.storage = StorageManager()
        self.matcher = DTWMatcher()
        self.visualizer = Visualizer()
        
        # Settings
        self.fps = 30
        self.frames_per_sign = 30

    def print_menu(self):
        print("\n--- Indian Sign Language (ISL) Recognition System ---")
        print("1. Train new sign (record A-Z)")
        print("2. List stored signs")
        print("3. Recognize from webcam (real-time)")
        print("4. Recognize from video file")
        print("5. Delete sign")
        print("6. Test accuracy")
        print("0. Exit")
        print("-----------------------------------------------------")

    def run(self):
        while True:
            self.print_menu()
            choice = input("Select an option: ")
            
            if choice == '1':
                letter = input("Enter letter to record (A-Z): ").upper()
                self._record_sign_flow(letter)
            elif choice == '2':
                self._list_signs()
            elif choice == '3':
                self._recognize_webcam()
            elif choice == '4':
                filepath = input("Enter video file path: ")
                if os.path.exists(filepath):
                    self._recognize_video(filepath)
                else:
                    print(f"File not found: {filepath}")
            elif choice == '5':
                letter = input("Enter letter to delete: ").upper()
                count = self.storage.delete_sign(letter)
                print(f"Deleted {count} versions of {letter}.")
            elif choice == '6':
                self._test_accuracy()
            elif choice == '0':
                print("Exiting...")
                break
            else:
                print("Invalid option.")

    def _record_sign_flow(self, letter):
        cap = cv2.VideoCapture(0)
        print("Preparing to record...")
        
        # Countdown
        for i in range(3, 0, -1):
            print(f"Starting in {i}...")
            time.sleep(1)
            
        print("Recording...")
        recorded_frames = []
        last_metadata = None
        
        while len(recorded_frames) < self.frames_per_sign:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1) # Mirror
            landmarks, hand_detected, results = self.extractor.extract_landmarks(frame)
            
            # Save frame data
            recorded_frames.append(landmarks)
            last_metadata = self.extractor.get_hand_features(landmarks, hand_detected)
            
            # Feedback
            annotated = self.visualizer.draw_landmarks(frame, results)
            cv2.putText(annotated, f"Recording {letter}: {len(recorded_frames)}/{self.frames_per_sign}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Sign Recording", annotated)
            cv2.waitKey(1)
            
        cap.release()
        cv2.destroyAllWindows()
        
        if len(recorded_frames) == self.frames_per_sign:
            sequence = np.array(recorded_frames)
            filepath = self.storage.save_sign(letter, sequence, last_metadata)
            print(f"Saved {letter} with {self.frames_per_sign} frames, 42 landmarks per frame.")
            print(f"File: {filepath}")
            
            # Show 3D skeleton of a sample
            self.visualizer.plot_3d_skeleton(sequence[self.frames_per_sign//2], title=f"Saved Sign: {letter}")
        else:
            print("Recording failed or interrupted.")

    def _list_signs(self):
        summary = self.storage.list_signs()
        if not summary:
            print("No signs recorded yet.")
        else:
            print("\nStored Signs:")
            for letter, count in sorted(summary.items()):
                print(f" - {letter}: {count} version(s)")

    def _recognize_webcam(self):
        print("Loading library...")
        library = self.storage.load_all_signs()
        if not library:
            print("No signs trained. Train signs first.")
            return
            
        cap = cv2.VideoCapture(0)
        buffer = []
        
        print("Show sign... Press 'q' to stop.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            landmarks, hand_detected, results = self.extractor.extract_landmarks(frame)
            buffer.append(landmarks)
            
            annotated = self.visualizer.draw_landmarks(frame, results)
            
            # Keep buffer size to self.frames_per_sign
            if len(buffer) > self.frames_per_sign:
                buffer.pop(0)
                
            if len(buffer) == self.frames_per_sign:
                # Every N frames or on a rolling basis; here we do it rolling
                sequence = np.array(buffer)
                matches = self.matcher.match(sequence, library, top_k=3)
                
                if matches:
                    best = matches[0]
                    # Update screen if confidence is decent
                    if best["confidence"] > 0:
                        cv2.putText(annotated, f"Detected: {best['letter']} ({best['confidence']*100:.0f}%)", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        y_pos = 70
                        for m in matches[1:3]:
                            cv2.putText(annotated, f"Alt: {m['letter']} ({m['confidence']*100:.0f}%)", 
                                        (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                            y_pos += 30
                            
            cv2.imshow("Sign Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def _recognize_video(self, filepath):
        print("Loading library...")
        library = self.storage.load_all_signs()
        if not library:
            print("No signs trained. Train signs first.")
            return
            
        cap = cv2.VideoCapture(filepath)
        frames_list = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            landmarks, _, _ = self.extractor.extract_landmarks(frame)
            frames_list.append(landmarks)
            
        cap.release()
        
        if len(frames_list) < self.frames_per_sign:
            print(f"Video too short. Needs at least {self.frames_per_sign} frames.")
            return
            
        # We'll test middle segment
        start_idx = max(0, (len(frames_list) - self.frames_per_sign) // 2)
        sequence = np.array(frames_list[start_idx:start_idx+self.frames_per_sign])
        
        matches = self.matcher.match(sequence, library, top_k=3)
        if matches:
            print(f"Top Match: {matches[0]['letter']} (Confidence: {matches[0]['confidence']*100:.1f}%)")
            print("Other matches:")
            for m in matches[1:]:
                print(f" - {m['letter']}: {m['confidence']*100:.1f}%")
        else:
            print("No valid matches found.")

    def _test_accuracy(self):
        """Record a known sign and check if it matches correctly."""
        print("--- Test Accuracy ---")
        expected_letter = input("What letter will you sign? ").upper()
        
        # Borrow recording logic
        cap = cv2.VideoCapture(0)
        for i in range(3, 0, -1):
            print(f"Starting test for {expected_letter} in {i}...")
            time.sleep(1)
            
        recorded_frames = []
        while len(recorded_frames) < self.frames_per_sign:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            landmarks, _, _ = self.extractor.extract_landmarks(frame)
            recorded_frames.append(landmarks)
            cv2.waitKey(1)
            
        cap.release()
        cv2.destroyAllWindows()
        
        if len(recorded_frames) == self.frames_per_sign:
            sequence = np.array(recorded_frames)
            library = self.storage.load_all_signs()
            matches = self.matcher.match(sequence, library, top_k=3)
            
            if matches:
                top_match = matches[0]['letter']
                success = (top_match == expected_letter)
                print(f"\nResult: {'SUCCESS' if success else 'FAILED'}")
                print(f"Expected: {expected_letter}, Got: {top_match} (Conf: {matches[0]['confidence']*100:.1f}%)")
                
                # Show comparison with the best match visually
                if success and library.get(top_match):
                    best_match_seq = library[top_match][0]["landmarks"]
                    self.visualizer.show_comparison(sequence, best_match_seq, top_match)
        else:
            print("Test recording failed.")

if __name__ == "__main__":
    app = ISLRecognizerApp()
    app.run()
