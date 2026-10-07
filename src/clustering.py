import os
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def compute_elbow(X_scaled, min_k=2, max_k=8):
    """
    Compute K-Means inertia for a range of k values (default 2 to 8).
    Returns pandas DataFrame with 'K' and 'Inertia'.
    """
    elbow_results = []
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        elbow_results.append({
            'K': k,
            'Inertia': kmeans.inertia_
        })
    return pd.DataFrame(elbow_results)


def compute_pca(X_scaled, n_components=2):
    """
    Reduce dimensionality of scaled features using PCA.
    Returns:
    - pca_df: DataFrame with PC1, PC2...
    - pca_model: Fitted PCA object
    - explained_variance_ratio: Variance explained per component
    - cumulative_variance: Cumulative variance explained
    """
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)
    
    col_names = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(components, columns=col_names)
    
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    return pca_df, pca, var_ratio, cum_var


def train_kmeans(X_scaled, n_clusters=4, random_state=42):
    """
    Train K-Means clustering model on scaled features.
    Returns fitted model and predicted cluster labels.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def save_ml_artifacts(models_dir, kmeans_model, scaler, pca_model, feature_names, cluster_profiles):
    """
    Save trained model artifacts into models directory using Joblib.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    joblib.dump(kmeans_model, os.path.join(models_dir, 'kmeans_model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(pca_model, os.path.join(models_dir, 'pca_model.pkl'))
    joblib.dump(feature_names, os.path.join(models_dir, 'feature_names.pkl'))
    joblib.dump(cluster_profiles, os.path.join(models_dir, 'cluster_profiles.pkl'))
    
    print(f"Artifacts saved successfully to {models_dir}")


def load_ml_artifacts(models_dir):
    """
    Load saved model artifacts from models directory.
    Returns dict of loaded artifacts or None if files don't exist.
    """
    required_files = [
        'kmeans_model.pkl', 'scaler.pkl', 'pca_model.pkl',
        'feature_names.pkl', 'cluster_profiles.pkl'
    ]
    
    artifacts = {}
    for fname in required_files:
        fpath = os.path.join(models_dir, fname)
        if os.path.exists(fpath):
            artifacts[fname.replace('.pkl', '')] = joblib.load(fpath)
        else:
            return None
            
    return artifacts
