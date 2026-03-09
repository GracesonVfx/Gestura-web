import numpy as np

class DTWMatcher:
    def __init__(self):
        pass
        
    def flatten_sequence(self, sequence):
        """
        Flattens a sequence of shape (N, 42, 3) to (N, 126)
        """
        n_frames = sequence.shape[0]
        return sequence.reshape(n_frames, -1)

    def calculate_distance(self, seq1, seq2):
        """
        Calculates DTW distance between two sequences.
        Both sequences are expected to be shape (N, 42, 3) and (M, 42, 3).
        Dynamic Time Warping finds the optimal alignment between two time series.
        """
        flat_seq1 = self.flatten_sequence(seq1)
        flat_seq2 = self.flatten_sequence(seq2)
        
        n, m = len(flat_seq1), len(flat_seq2)
        
        # Create cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Build cost matrix using Euclidean distance
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Euclidean distance between frames
                cost = np.linalg.norm(flat_seq1[i-1] - flat_seq2[j-1])
                
                # Take minimum cost path
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],    # insertion
                    dtw_matrix[i, j-1],    # deletion
                    dtw_matrix[i-1, j-1]   # match
                )
                
        # The DTW distance is the value at the bottom right corner
        # Normalize by the sum of sequence lengths to balance long vs short paths
        normalized_distance = dtw_matrix[n, m] / (n + m)
        return normalized_distance

    def match(self, input_sequence, library_signs, top_k=3):
        """
        Matches an input sequence against a library of stored signs.
        library_signs is a dict: { 'A': [ {landmarks: ...}, ... ], 'B': ... }
        Returns top k matches and their confidence scores.
        """
        results = []
        
        for letter, datasets in library_signs.items():
            for idx, data in enumerate(datasets):
                stored_seq = data["landmarks"]
                
                # Check shapes to prevent errors
                if input_sequence.shape[1:] != stored_seq.shape[1:]:
                    continue
                    
                distance = self.calculate_distance(input_sequence, stored_seq)
                
                # Convert distance to confidence score: 1 / (1 + distance)
                confidence = 1.0 / (1.0 + distance)
                
                results.append({
                    "letter": letter,
                    "confidence": confidence,
                    "distance": distance,
                    "version": data.get("version", 1)
                })
                
        # Sort by confidence descending
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Deduplicate to just get top letters (best version of each)
        best_matches = []
        seen_letters = set()
        
        for res in results:
            if res["letter"] not in seen_letters:
                best_matches.append(res)
                seen_letters.add(res["letter"])
                if len(best_matches) >= top_k:
                    break
                    
        return best_matches
