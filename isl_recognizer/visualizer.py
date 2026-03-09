import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

    def draw_landmarks(self, frame, results):
        """
        Draws MediaPipe landmarks on a given RGB frame.
        """
        annotated_image = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
        return annotated_image

    def plot_3d_skeleton(self, landmarks_array, title="3D Hand Skeleton"):
        """
        Plots a 3D scatter plot of the extracted landmarks.
        landmarks_array: shape (42, 3), where first 21 are right hand, next 21 are left.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(title)
        
        # Right hand (Indices 0-20)
        right_points = landmarks_array[0:21]
        if np.any(right_points): # Check if hand is present
            xs_r = right_points[:, 0]
            ys_r = right_points[:, 1]
            zs_r = right_points[:, 2]
            ax.scatter(xs_r, zs_r, -ys_r, c='b', marker='o', label='Right Hand')
            
            # Draw connections (using MediaPipe's HAND_CONNECTIONS)
            for connection in self.mp_hands.HAND_CONNECTIONS:
                idx1, idx2 = connection
                ax.plot([xs_r[idx1], xs_r[idx2]], [zs_r[idx1], zs_r[idx2]], [-ys_r[idx1], -ys_r[idx2]], 'b-')

        # Left hand (Indices 21-41)
        left_points = landmarks_array[21:42]
        if np.any(left_points):
            xs_l = left_points[:, 0]
            ys_l = left_points[:, 1]
            zs_l = left_points[:, 2]
            ax.scatter(xs_l, zs_l, -ys_l, c='r', marker='^', label='Left Hand')
            
            for connection in self.mp_hands.HAND_CONNECTIONS:
                idx1, idx2 = connection
                ax.plot([xs_l[idx1], xs_l[idx2]], [zs_l[idx1], zs_l[idx2]], [-ys_l[idx1], -ys_l[idx2]], 'r-')

        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (inverted)')
        ax.legend()
        plt.show()

    def show_comparison(self, input_sequence, match_sequence, letter):
        """
        Plots the average or middle frame pose of input vs matched.
        """
        # Taking the middle frame for a quick visual comparison
        mid_idx = len(input_sequence) // 2
        input_frame = input_sequence[mid_idx]
        match_frame = match_sequence[mid_idx]
        
        fig = plt.figure(figsize=(12, 6))
        
        # Plot Input
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.set_title("Your Input Sign")
        self._plot_single_sequence(ax1, input_frame)
        
        # Plot Match
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_title(f"Best Match ({letter})")
        self._plot_single_sequence(ax2, match_frame)
        
        plt.tight_layout()
        plt.show()

    def _plot_single_sequence(self, ax, landmarks_array):
        # Internal helper to plot points without blocking or creating new figures
        right_points = landmarks_array[0:21]
        if np.any(right_points):
            xs_r, ys_r, zs_r = right_points[:, 0], right_points[:, 1], right_points[:, 2]
            ax.scatter(xs_r, zs_r, -ys_r, c='b')
            for connection in self.mp_hands.HAND_CONNECTIONS:
                i1, i2 = connection
                ax.plot([xs_r[i1], xs_r[i2]], [zs_r[i1], zs_r[i2]], [-ys_r[i1], -ys_r[i2]], 'b-')
                
        left_points = landmarks_array[21:42]
        if np.any(left_points):
            xs_l, ys_l, zs_l = left_points[:, 0], left_points[:, 1], left_points[:, 2]
            ax.scatter(xs_l, zs_l, -ys_l, c='r')
            for connection in self.mp_hands.HAND_CONNECTIONS:
                i1, i2 = connection
                ax.plot([xs_l[i1], xs_l[i2]], [zs_l[i1], zs_l[i2]], [-ys_l[i1], -ys_l[i2]], 'r-')
                
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y (inv)')
