import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_dataset(filepath_or_buffer):
    """
    Load dataset from CSV file or buffer.
    """
    if isinstance(filepath_or_buffer, str):
        df = pd.read_csv(filepath_or_buffer)
    else:
        df = pd.read_csv(filepath_or_buffer)
    return df


def detect_columns(df):
    """
    Automatically inspect and detect column roles:
    - Target/Activity column
    - ID/Subject columns
    - Sensor numerical columns
    - Categorical columns
    """
    target_col = None
    id_cols = []
    
    # Common target column names
    target_keywords = ['activity', 'label', 'target', 'class', 'act_name', 'activity_label']
    for col in df.columns:
        if col.strip().lower() in target_keywords:
            target_col = col
            break

    # If not matched directly, check if a text column with 3-10 unique values exists
    if target_col is None:
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            if 2 <= df[col].nunique() <= 15:
                target_col = col
                break

    # Detect ID columns
    id_keywords = ['subject', 'id', 'user', 'user_id', 'subject_id', 'index', 'unnamed: 0']
    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in id_keywords or col_lower.startswith('id_') or col_lower.endswith('_id'):
            if col != target_col:
                id_cols.append(col)

    # Numerical & Categorical identification
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()

    # Sensor features are numerical columns excluding ID cols and target col
    sensor_cols = [c for c in num_cols if c not in id_cols and c != target_col]

    return {
        'target_col': target_col,
        'id_cols': id_cols,
        'sensor_cols': sensor_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols
    }


def preprocess_data(df, target_col=None, id_cols=None):
    """
    Preprocess dataset:
    - Remove duplicate records
    - Handle infinite / invalid values
    - Impute missing numerical values using median
    - Drop single-value (constant) columns
    Returns cleaned dataframe, clean numerical feature matrix X, and summary stats dictionary.
    """
    df_clean = df.copy()
    initial_shape = df_clean.shape

    # 1. Remove duplicate records
    duplicate_count = df_clean.duplicated().sum()
    if duplicate_count > 0:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Automatically detect columns if not provided
    detected = detect_columns(df_clean)
    if target_col is None:
        target_col = detected['target_col']
    if id_cols is None:
        id_cols = detected['id_cols']

    # Identify candidate sensor numerical columns
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    sensor_cols = [c for c in num_cols if c not in id_cols and c != target_col]

    # 2. Convert sensor columns to numeric if needed, coerce errors
    for col in sensor_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 3. Detect and replace infinite values with NaN
    df_clean[sensor_cols] = df_clean[sensor_cols].replace([np.inf, -np.inf], np.nan)

    # 4. Check missing values count
    missing_count = df_clean[sensor_cols].isnull().sum().sum()

    # 5. Impute missing numerical values using Median Imputer
    if missing_count > 0:
        imputer = SimpleImputer(strategy='median')
        df_clean[sensor_cols] = imputer.fit_transform(df_clean[sensor_cols])

    # 6. Remove constant columns (columns with only 1 unique value)
    constant_cols = [c for c in sensor_cols if df_clean[c].nunique() <= 1]
    if constant_cols:
        df_clean = df_clean.drop(columns=constant_cols)
        sensor_cols = [c for c in sensor_cols if c not in constant_cols]

    # Feature matrix X
    X = df_clean[sensor_cols].copy()

    summary = {
        'initial_shape': initial_shape,
        'cleaned_shape': df_clean.shape,
        'duplicate_count': duplicate_count,
        'missing_values_imputed': int(missing_count),
        'dropped_constant_cols': constant_cols,
        'target_col': target_col,
        'id_cols': id_cols,
        'sensor_cols': sensor_cols
    }

    return df_clean, X, summary


def scale_features(X):
    """
    Standardize feature matrix using StandardScaler.
    Returns scaled array X_scaled and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
