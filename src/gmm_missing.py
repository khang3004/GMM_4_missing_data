import numpy as np
from scipy.stats import multivariate_normal
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import KMeans
from scipy.special import logsumexp

class GMMMissing(BaseEstimator, ClusterMixin):
    """
    Gaussian Mixture Model Clustering with Incomplete Data.
    Follows the algorithm proposed in:
    "Gaussian Mixture Model Clustering with Incomplete Data", Zhang et al.
    """
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-4, 
                 reg_covar=1e-6, random_state=None, init_params='kmeans'):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state
        self.init_params = init_params
        
    def _initialize_parameters(self, X):
        """Initialize GMM parameters."""
        n_samples, n_features = X.shape
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        if self.init_params == 'kmeans':
            kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
            labels = kmeans.fit_predict(X)
            self.mu_ = kmeans.cluster_centers_
        else:
            random_indices = np.random.choice(n_samples, self.n_components, replace=False)
            self.mu_ = X[random_indices].copy()
            labels = np.random.randint(0, self.n_components, n_samples)
            
        self.alpha_ = np.ones(self.n_components) / self.n_components
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
        
        # Estimate initial covariances and alpha based on labels
        for k in range(self.n_components):
            mask_k = (labels == k)
            if np.sum(mask_k) > 0:
                self.alpha_[k] = np.mean(mask_k)
                X_k = X[mask_k]
                cov_k = np.cov(X_k.T)
                # Ensure it's not singular by adding reg_covar
                cov_k += np.eye(n_features) * self.reg_covar
                self.covariances_[k] = cov_k
                
    def _e_step(self, X):
        """
        Expectation step.
        Computes the log-likelihood and the responsibilities (gamma).
        """
        n_samples, n_features = X.shape
        log_resp = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Calculate log pdf for component k
            try:
                rv = multivariate_normal(mean=self.mu_[k], cov=self.covariances_[k], allow_singular=True)
                log_prob = rv.logpdf(X)
                log_resp[:, k] = np.log(self.alpha_[k] + 1e-300) + log_prob
            except np.linalg.LinAlgError:
                # Fallback if covariance gets severely singular despite reg_covar
                pseudo_cov = self.covariances_[k] + np.eye(n_features) * (self.reg_covar * 10)
                rv = multivariate_normal(mean=self.mu_[k], cov=pseudo_cov, allow_singular=True)
                log_prob = rv.logpdf(X)
                log_resp[:, k] = np.log(self.alpha_[k] + 1e-300) + log_prob
        
        log_prob_norm = logsumexp(log_resp, axis=1)
        log_resp -= log_prob_norm[:, np.newaxis]
        gamma = np.exp(log_resp)
        
        return log_prob_norm.sum(), gamma
        
    def _m_step_params(self, X, gamma):
        """
        Maximization step for the parameters (alpha, mu, Sigma).
        Equations 9, 10, 11
        """
        n_samples, n_features = X.shape
        N_k = gamma.sum(axis=0)
        
        # Update alpha
        self.alpha_ = N_k / n_samples
        
        for k in range(self.n_components):
            # Update mu
            if N_k[k] > 1e-15:
                # gamma[:, k] is (n_samples,) and X is (n_samples, n_features)
                self.mu_[k] = (gamma[:, k][:, np.newaxis] * X).sum(axis=0) / N_k[k]
                
                # Update Covariance
                diff = X - self.mu_[k]
                cov_k = (gamma[:, k][:, np.newaxis, np.newaxis] * (diff[:, :, np.newaxis] @ diff[:, np.newaxis, :])).sum(axis=0) / N_k[k]
                self.covariances_[k] = cov_k + np.eye(n_features) * self.reg_covar

    def _m_step_data(self, X, mask, gamma):
        """
        Maximization step for the missing data (X_m).
        Equation 14.
        """
        n_samples, n_features = X.shape
        X_updated = np.copy(X)
        
        # Precompute precision matrices for each component
        precisions = np.zeros_like(self.covariances_)
        for k in range(self.n_components):
            precisions[k] = np.linalg.inv(self.covariances_[k])
            
        for i in range(n_samples):
            # Boolean mask for the current sample: True means missing (m), False means observed (o)
            m_part = mask[i]
            if not np.any(m_part):
                continue
                
            o_part = ~m_part
            
            # Subsets
            dim_m = np.sum(m_part)
            x_i_o = X[i, o_part]
            
            sum_L_mm = np.zeros((dim_m, dim_m))
            sum_term2 = np.zeros(dim_m)
            
            for k in range(self.n_components):
                gamma_ik = gamma[i, k]
                if gamma_ik < 1e-15:
                    continue
                    
                Lambda_k = precisions[k]
                
                L_mm = Lambda_k[np.ix_(m_part, m_part)] # Sigma_{imm}^{-1}
                L_mo = Lambda_k[np.ix_(m_part, o_part)] # Sigma_{imo}^{-1}
                
                mu_m = self.mu_[k, m_part]
                mu_o = self.mu_[k, o_part]
                
                sum_L_mm += gamma_ik * L_mm
                term2 = L_mm @ mu_m - L_mo @ (x_i_o - mu_o)
                sum_term2 += gamma_ik * term2
                
            try:
                x_i_m_updated = np.linalg.inv(sum_L_mm) @ sum_term2
                X_updated[i, m_part] = x_i_m_updated
            except np.linalg.LinAlgError:
                # Fallback if precision submatrix is singular for some reason
                X_updated[i, m_part] = X[i, m_part]
                
        return X_updated

    def fit(self, X_incomplete):
        """
        Alternating EM Optimization unifying imputation and clustering.
        """
        # Create missing flag mask
        self.mask_ = np.isnan(X_incomplete)
        
        # Initial Imputation to start the algorithm (Mean filling)
        X = np.copy(X_incomplete)
        col_means = np.nanmean(X, axis=0)
        # If an entire column is NaN, fill with 0 to prevent issues
        col_means = np.nan_to_num(col_means)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
        
        self._initialize_parameters(X)
        self.lower_bound_ = -np.inf
        
        for iteration in range(self.max_iter):
            prev_lower_bound = self.lower_bound_
            
            # E-step
            log_prob_norm, gamma = self._e_step(X)
            self.lower_bound_ = log_prob_norm
            
            # Check for convergence
            change = self.lower_bound_ - prev_lower_bound
            if abs(change) < self.tol:
                break
                
            # M-step: parameters
            self._m_step_params(X, gamma)
            
            # M-step: update missing data points
            X = self._m_step_data(X, self.mask_, gamma)
            
        self.X_final_ = X
        self.n_iter_ = iteration
        return self

    def predict(self, X=None):
        """Predict labels for final dataset after fit or a new dataset."""
        if X is None:
            # Predict for the training set (from the final imputed version)
            _, gamma = self._e_step(self.X_final_)
        else:
            _, gamma = self._e_step(X)
        return np.argmax(gamma, axis=1)

    def predict_proba(self, X=None):
        """Predict probabilities (gamma) for final dataset or a new dataset."""
        if X is None:
            _, gamma = self._e_step(self.X_final_)
        else:
            _, gamma = self._e_step(X)
        return gamma
