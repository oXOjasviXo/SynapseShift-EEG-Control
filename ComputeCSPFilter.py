import os
import pickle
import numpy as np
import scipy.linalg as la

def compute_avg_covariance(X):
    """
    Computes the average covariance matrix for a list/array of samples.
    Each sample is assumed to have shape (channels, time_samples).
    """
    covs = [np.cov(sample) for sample in X]
    return np.mean(covs, axis=0)

def compute_csp_filters(X_raw, y, target_label, n_filters=2):
    """
    Computes CSP filters to differentiate between target (motor) and non-target (non-motor) samples.
    
    Parameters:
      X_raw : np.array
          Raw EEG data samples of shape (N_samples, channels, time_samples)
      y : np.array
          Labels for each sample.
      target_label : int
          The label corresponding to the "motor" condition.
      n_filters : int
          Number of spatial filters to select from each extreme (total filters will be 2*n_filters).
    
    Returns:
      filters : np.array
          CSP filters with shape (2*n_filters, channels)
    """
    # Separate samples for target (motor) and non-target (rest)
    X_target = X_raw[y == target_label]
    X_non_target = X_raw[y != target_label]
    
    cov_target = compute_avg_covariance(X_target)
    cov_non_target = compute_avg_covariance(X_non_target)
    
    # Composite covariance matrix
    composite_cov = cov_target + cov_non_target
    
    # Solve the generalized eigenvalue problem: cov_target * v = lambda * composite_cov * v
    eigvals, eigvecs = la.eigh(cov_target, composite_cov)
    
    # Sort eigenvalues in descending order and get corresponding eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sorted_indices]
    
    # Select the top n_filters and bottom n_filters eigenvectors
    filters = np.concatenate((eigvecs[:, :n_filters], eigvecs[:, -n_filters:]), axis=1)
    
    # Transpose so that the output shape is (2*n_filters, channels)
    return filters.T

def main():
    # Path to the calibration data file
    data_file = "data/calibration_data.pkl"

    with open(data_file, "rb") as f:
        data = pickle.load(f)
    
    # X_raw is an array of raw EEG samples with shape (N_samples, 8, 125)
    X_raw = np.array(data["features"])
    y = np.array(data["labels"])
    
    # Define the target label for motor. Adjust this based on label mapping.
    # ACTIONS = {"none": 0, "blink": 1, "jaw": 2, "relax": 3, "motor": 4, "mental": 5}
    # then target_label for "motor" should be 4.
    target_label = 4
    
    # Compute CSP filters (using n_filters from each side, total 2*n_filters filters)
    try:
        filters = compute_csp_filters(X_raw, y, target_label, n_filters=2)
    except Exception as e:
        print("Error computing CSP filters:", e)
        return
    
    # Save the CSP filters to a file for later use during training/running the system
    with open("csp_filters.pkl", "wb") as f:
        pickle.dump(filters, f)
    print("CSP filters computed and saved to 'csp_filters.pkl'.")

if __name__ == "__main__":
    main()