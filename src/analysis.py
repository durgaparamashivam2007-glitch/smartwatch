import numpy as np
import pandas as pd


def determine_activity_level(dominant_activity, sensor_energy=None):
    """
    Map an activity name or sensor energy to an Activity Level category.
    """
    if pd.isna(dominant_activity) or dominant_activity is None or str(dominant_activity).lower() == 'n/a':
        if sensor_energy is not None:
            if sensor_energy > 0.6:
                return 'High'
            elif sensor_energy > 0.2:
                return 'Moderate'
            elif sensor_energy > -0.2:
                return 'Light'
            else:
                return 'Low'
        return 'Moderate'

    act_str = str(dominant_activity).upper()
    if 'UPSTAIRS' in act_str or 'RUNNING' in act_str or 'VIGOROUS' in act_str:
        return 'High'
    elif 'DOWNSTAIRS' in act_str or 'WALKING' in act_str or 'DYNAMIC' in act_str:
        return 'Moderate to High'
    elif 'STANDING' in act_str or 'LIGHT' in act_str:
        return 'Light'
    elif 'SITTING' in act_str or 'LAYING' in act_str or 'REST' in act_str or 'STATIONARY' in act_str:
        return 'Low / Sedentary'
    else:
        return 'Moderate'


def determine_behavior_group(dominant_activity, cluster_id, mean_sensor_mag=0.0):
    """
    Data-driven behavior group naming based on dominant activity label and signal intensity.
    Possible behavior group names:
    - Highly Active
    - Dynamic Movement
    - Moderately Active
    - Light Activity
    - Sedentary/Stationary
    """
    if dominant_activity and str(dominant_activity).lower() != 'n/a':
        act_str = str(dominant_activity).upper()
        if 'WALKING_UPSTAIRS' in act_str:
            return f"Highly Active ({dominant_activity.capitalize()})"
        elif 'WALKING_DOWNSTAIRS' in act_str:
            return f"Dynamic Movement ({dominant_activity.capitalize()})"
        elif 'WALKING' in act_str:
            return f"Dynamic Movement ({dominant_activity.capitalize()})"
        elif 'STANDING' in act_str:
            return f"Light Activity ({dominant_activity.capitalize()})"
        elif 'SITTING' in act_str:
            return f"Sedentary/Stationary ({dominant_activity.capitalize()})"
        elif 'LAYING' in act_str:
            return f"Sedentary/Stationary ({dominant_activity.capitalize()})"
        else:
            return f"Behavior Group {cluster_id} ({dominant_activity})"
    else:
        # Fallback based on sensor signal intensity relative thresholding
        if mean_sensor_mag > 0.5:
            return f"Highly Active (Cluster {cluster_id})"
        elif mean_sensor_mag > 0.1:
            return f"Dynamic Movement (Cluster {cluster_id})"
        elif mean_sensor_mag > -0.2:
            return f"Moderately Active (Cluster {cluster_id})"
        else:
            return f"Sedentary/Stationary (Cluster {cluster_id})"


def profile_clusters(df, cluster_col, feature_cols, target_col=None):
    """
    Generates detailed cluster profiling stats:
    - Cluster ID
    - Record count & percentage
    - Dominant activity label (if available)
    - Data-driven Behavior Group Name
    - Activity Level
    - Mean sensor readings for selected features
    """
    clusters = sorted(df[cluster_col].unique())
    profiles = []

    # Calculate overall sensor signal magnitude proxy (mean of feature means)
    global_sensor_mean = df[feature_cols].mean(axis=1)
    df_temp = df.copy()
    df_temp['_signal_mag'] = global_sensor_mean

    for c in clusters:
        sub = df_temp[df_temp[cluster_col] == c]
        records_count = len(sub)
        pct = (records_count / len(df)) * 100

        dominant_act = 'N/A'
        dominant_act_pct = 0.0

        if target_col and target_col in df.columns:
            act_counts = sub[target_col].value_counts()
            if not act_counts.empty:
                dominant_act = act_counts.index[0]
                dominant_act_pct = (act_counts.iloc[0] / records_count) * 100

        mean_mag = sub['_signal_mag'].mean()
        behavior_group = determine_behavior_group(dominant_act, c, mean_sensor_mag=mean_mag)
        act_level = determine_activity_level(dominant_act, sensor_energy=mean_mag)

        # Average sensor measurements (top features)
        top_feat_means = sub[feature_cols[:10]].mean().to_dict()

        profile = {
            'Cluster': f"Cluster {c}",
            'Cluster_ID': int(c),
            'Behavior Group': behavior_group,
            'Records': records_count,
            'Percentage (%)': round(pct, 2),
            'Dominant Activity': dominant_act,
            'Dominant Activity Share (%)': round(dominant_act_pct, 2),
            'Activity Level': act_level,
            'Mean Signal Intensity': round(mean_mag, 4)
        }

        # Include sample sensor means in profile
        for f_name, f_val in top_feat_means.items():
            profile[f"Mean_{f_name}"] = round(float(f_val), 4)

        profiles.append(profile)

    profile_df = pd.DataFrame(profiles)
    return profile_df


def select_important_features(X, n_features=6):
    """
    Select top n_features based on variance across samples.
    Provides representative inputs for the prediction interface.
    """
    variances = X.var(axis=0)
    top_indices = variances.nlargest(n_features).index.tolist()
    return top_indices
