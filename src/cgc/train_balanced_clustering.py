import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm

class BalancedKMeans:
    def __init__(
        self, 
        n_clusters: int, 
        max_iters: int = 100,
        tolerance: float = 1e-4,
        verbose: bool = False
    ):
        """
        Initialize Balanced K-means clustering algorithm.
        
        Args:
            n_clusters: Number of clusters
            max_iters: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Whether to show progress bar
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tolerance = tolerance
        self.verbose = verbose
        self.centroids = None
        self.labels = None
        
    def fit(self, X: np.ndarray) -> 'BalancedKMeans':
        """
        Fit balanced k-means clustering.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            self: Fitted clustering object
        """
        n_samples = len(X)
        # Compute cluster size (m in the algorithm)
        self.cluster_size = n_samples // self.n_clusters
        
        # Randomly initialize centroids
        idx = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        # Initialize iteration counter
        iter_count = 0
        prev_centroids = None
        
        iterator = range(self.max_iters)
        if self.verbose:
            iterator = tqdm(iterator, desc="Balanced K-means iterations")
            
        for _ in iterator:
            # Compute minimum distances from points to centroids
            distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
            min_distances = np.min(distances, axis=1)
            
            # Sort points by minimum distance
            sorted_indices = np.argsort(min_distances)
            X_sorted = X[sorted_indices]
            
            # Initialize cluster assignments
            self.labels = np.full(n_samples, -1)
            cluster_counts = np.zeros(self.n_clusters, dtype=int)
            
            # Process points in order of increasing minimum distance
            for i in range(n_samples):
                point = X_sorted[i]
                point_distances = np.linalg.norm(point - self.centroids, axis=1)
                
                # Find nearest cluster that isn't full
                sorted_cluster_indices = np.argsort(point_distances)
                assigned = False
                
                # Try to assign to the nearest non-full cluster
                for cluster_idx in sorted_cluster_indices:
                    if cluster_counts[cluster_idx] < self.cluster_size:
                        self.labels[sorted_indices[i]] = cluster_idx
                        cluster_counts[cluster_idx] += 1
                        assigned = True
                        break
                        
                # If all clusters are full (shouldn't happen with balanced sizes)
                if not assigned:
                    self.labels[sorted_indices[i]] = sorted_cluster_indices[0]
                    cluster_counts[sorted_cluster_indices[0]] += 1
                    
            # Update centroids
            prev_centroids = self.centroids.copy()
            for j in range(self.n_clusters):
                cluster_points = X[self.labels == j]
                if len(cluster_points) > 0:
                    self.centroids[j] = np.mean(cluster_points, axis=0)
                    
            # Check convergence
            if prev_centroids is not None:
                centroid_shift = np.max(np.linalg.norm(self.centroids - prev_centroids, axis=1))
                if centroid_shift < self.tolerance:
                    if self.verbose:
                        print(f"Converged after {iter_count + 1} iterations")
                    break
                    
            iter_count += 1
            
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            labels: Predicted cluster labels
        """
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit clustering and predict cluster labels.
        
        Args:
            X: Input array of shape (n_samples, n_features)
            
        Returns:
            labels: Predicted cluster labels
        """
        self.fit(X)
        return self.labels
    
    def rearrange_codebook(self, X: np.ndarray) -> np.ndarray:
        """
        Rearrange the codebook based on cluster assignments.
        Following the algorithm, assigns embeddings in cluster j to indices [jm, (j+1)m)
        where m is the cluster size.
        
        Args:
            X: Input codebook array of shape (n_samples, n_features)
            
        Returns:
            X_rearranged: Rearranged codebook
        """
        self.fit(X)
        X_rearranged = np.zeros_like(X)
        m = self.cluster_size  # cluster size
        
        for j in range(self.n_clusters):
            cluster_points = X[self.labels == j]
            start_idx = j * m
            end_idx = (j + 1) * m
            # Ensure we don't exceed array bounds if cluster is not exactly size m
            actual_size = min(len(cluster_points), m)
            X_rearranged[start_idx:start_idx + actual_size] = cluster_points[:actual_size]
            
        return X_rearranged

# Example usage:
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(1000, 64)  # Example codebook with 1000 codes of dimension 64
    
    # Initialize and fit balanced k-means
    clustering = BalancedKMeans(n_clusters=10, verbose=True)
    
    # Rearrange codebook
    X_rearranged = clustering.rearrange_codebook(X)
    
    # Print cluster sizes
    unique, counts = np.unique(clustering.labels, return_counts=True)
    print("\nCluster sizes:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster}: {count}")