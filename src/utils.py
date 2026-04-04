import numpy as np
from sklearn.datasets import make_blobs

def generate_synthetic_data(n_samples=300, n_features=2, centers=3, cluster_std=1.0, 
                            missing_ratio=0.2, random_state=42):
    """
    Generates synthetic cluster data and injects Missing Completely At Random (MCAR) indices.
    
    Returns:
        X_true (np.ndarray): The ground truth complete data.
        X_incomplete (np.ndarray): The data with NaNs representing missing values.
        y (np.ndarray): The ground truth cluster labels.
        mask (np.ndarray): Boolean mask where True indicates a missing value.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    X_true, y = make_blobs(n_samples=n_samples, n_features=n_features, 
                           centers=centers, cluster_std=cluster_std, 
                           random_state=random_state)
    
    # Introduce missing values MCAR
    X_incomplete = np.copy(X_true)
    mask = np.random.rand(*X_true.shape) < missing_ratio
    
    # Ensure no row is entirely missing (for simplicity and stability)
    all_missing_rows = np.all(mask, axis=1)
    for row_idx in np.where(all_missing_rows)[0]:
        keep_col = np.random.randint(0, n_features)
        mask[row_idx, keep_col] = False
        
    X_incomplete[mask] = np.nan
    
    return X_true, X_incomplete, y, mask

def zero_imputation(X_incomplete):
    """
    Replaces NaNs with zeros.
    
    Returns:
        X_imputed (np.ndarray): Data with zeros replacing missing values.
    """
    X_imputed = np.copy(X_incomplete)
    X_imputed[np.isnan(X_imputed)] = 0.0
    return X_imputed

def mean_imputation(X_incomplete):
    """
    Replaces NaNs with the mean of their respective columns.
    
    Returns:
        X_imputed (np.ndarray): Data with mean-imputed values.
    """
    X_imputed = np.copy(X_incomplete)
    col_means = np.nanmean(X_imputed, axis=0)
    
    inds = np.where(np.isnan(X_imputed))
    X_imputed[inds] = np.take(col_means, inds[1])
    return X_imputed
