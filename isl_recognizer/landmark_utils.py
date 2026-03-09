import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_landmarks(self, frame):
        """
        Extracts 42 landmarks (21 per hand) from a frame.
        Always returns shape (42, 3), where the first 21 points belong to the right
        hand and the next 21 belong to the left hand.
        If a hand is missing, its points are zeroed.
        """
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        # 42 landmarks, 3 coordinates (x, y, z)
        landmarks_array = np.zeros((42, 3), dtype=np.float32)
        hand_detected = {"Right": False, "Left": False}
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # MediaPipe handedness label ('Left' or 'Right') - note standard mirror conventions
                label = handedness.classification[0].label
                
                # We'll assign index 0-20 for Right hand, 21-41 for Left
                offset = 0 if label == 'Right' else 21
                hand_detected[label] = True
                
                for i, lm in enumerate(hand_landmarks.landmark):
                    landmarks_array[offset + i] = [lm.x, lm.y, lm.z]
                    
        # Normalize the landmarks
        normalized_array = self.normalize_landmarks(landmarks_array, hand_detected)
        
        return normalized_array, hand_detected, results

    def normalize_landmarks(self, landmarks, hand_detected):
        """
        Normalizes landmarks by subtracting wrist coordinate and scaling by palm size.
        """
        normalized_landmarks = np.zeros_like(landmarks)
        
        # Right Hand Normalization (indices 0-20)
        if hand_detected["Right"]:
            wrist_r = landmarks[0]
            mcp_r = landmarks[9] # Middle finger MCP
            palm_size_r = np.linalg.norm(mcp_r - wrist_r)
            if palm_size_r > 0:
                normalized_landmarks[0:21] = (landmarks[0:21] - wrist_r) / palm_size_r
                
        # Left Hand Normalization (indices 21-41)
        if hand_detected["Left"]:
            wrist_l = landmarks[21]
            mcp_l = landmarks[30] # Middle finger MCP for left (21 + 9)
            palm_size_l = np.linalg.norm(mcp_l - wrist_l)
            if palm_size_l > 0:
                normalized_landmarks[21:42] = (landmarks[21:42] - wrist_l) / palm_size_l
                
        return normalized_landmarks

    def get_hand_features(self, landmarks_array, hand_detected):
        """
        Calculates ISL specific metadata such as dominant hand and finger states.
        """
        # Determine dominant hand based on which hand or which has more motion (basic heuristics)
        used_hands = [k for k, v in hand_detected.items() if v]
        which_hand_dominant = "both" if len(used_hands) == 2 else (used_hands[0] if used_hands else "none")
        
        finger_states = []
        # Right Hand
        if hand_detected["Right"]:
            # Simple heuristic for finger extension: finger tip y < finger pip y
            finger_states.extend(self._get_finger_states(landmarks_array[0:21]))
        else:
            finger_states.extend(["closed"] * 5)
            
        # Left Hand
        if hand_detected["Left"]:
            finger_states.extend(self._get_finger_states(landmarks_array[21:42]))
        else:
            finger_states.extend(["closed"] * 5)
            
        # Symmetry check (if both hands present, check if roughly mirror images)
        hand_symmetry = False
        if hand_detected["Right"] and hand_detected["Left"]:
            # Compare normalized poses (invert X for left hand to check symmetry)
            right_pose = landmarks_array[0:21].copy()
            left_pose = landmarks_array[21:42].copy()
            left_pose[:, 0] = -left_pose[:, 0] # Mirror x
            dist = np.linalg.norm(right_pose - left_pose)
            hand_symmetry = dist < 2.0 # Threshold for symmetry
            
        return {
            "which_hand_dominant": which_hand_dominant,
            "hand_symmetry": hand_symmetry,
            "finger_states": finger_states
        }

    def _get_finger_states(self, hand_landmarks):
        """Gets basic state of 5 fingers (extended or curled)."""
        states = []
        # Thumb, Index, Middle, Ring, Pinky tips: [4, 8, 12, 16, 20]
        # MCPs / PIPs: [2, 6, 10, 14, 18]
        # Basic heuristic: tip is higher (lower Y in image coords) than PIP/MCP
        tips = [4, 8, 12, 16, 20]
        pips = [2, 6, 10, 14, 18]
        
        for tip, pip in zip(tips, pips):
            # Since coordinate system origin is top-left, lower Y means higher position physically
            if hand_landmarks[tip, 1] < hand_landmarks[pip, 1]:
                states.append("extended")
            else:
                states.append("curled")
        return states
