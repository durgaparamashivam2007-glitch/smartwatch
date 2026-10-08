import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


# Custom Color Palette
COLOR_PALETTE = ['#00F2FE', '#4FACFE', '#00E676', '#FF9100', '#E040FB', '#FF5252', '#7C4DFF', '#FFEA00']
DARK_TEMPLATE = 'plotly_dark'


def plot_elbow_curve(elbow_df, selected_k=4):
    """
    Generate an interactive Plotly Elbow Method curve.
    """
    fig = go.Figure()

    # Inertia line
    fig.add_trace(go.Scatter(
        x=elbow_df['K'],
        y=elbow_df['Inertia'],
        mode='lines+markers',
        name='Inertia',
        line=dict(color='#00F2FE', width=3),
        marker=dict(size=10, color='#4FACFE', symbol='circle'),
        hovertemplate='<b>K = %{x}</b><br>Inertia = %{y:,.2f}<extra></extra>'
    ))

    # Highlight selected K
    if selected_k in elbow_df['K'].values:
        sel_row = elbow_df[elbow_df['K'] == selected_k].iloc[0]
        fig.add_trace(go.Scatter(
            x=[sel_row['K']],
            y=[sel_row['Inertia']],
            mode='markers',
            name=f'Selected K = {selected_k}',
            marker=dict(size=16, color='#FF5252', symbol='star'),
            hovertemplate=f'<b>Selected K = {selected_k}</b><br>Inertia = {sel_row["Inertia"]:,.2f}<extra></extra>'
        ))

    fig.update_layout(
        title=dict(text='<b>Elbow Method for Optimal K Selection</b>', font=dict(size=18)),
        xaxis=dict(title='Number of Clusters (K)', tickmode='linear', tick0=2, dtick=1),
        yaxis=dict(title='K-Means Inertia (Sum of Squared Distances)'),
        template=DARK_TEMPLATE,
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    return fig


def plot_pca_2d(pca_df, cluster_labels, target_labels=None, hover_df=None):
    """
    Generate interactive 2D PCA Scatter plot showing cluster groupings.
    """
    df_plot = pca_df.copy()
    df_plot['Cluster'] = [f'Cluster {c}' for c in cluster_labels]
    
    hover_cols = []
    if target_labels is not None:
        df_plot['Activity'] = target_labels.values if isinstance(target_labels, pd.Series) else target_labels
        hover_cols.append('Activity')

    if hover_df is not None and 'Behavior Group' in hover_df.columns:
        df_plot['Behavior Group'] = hover_df['Behavior Group'].values
        hover_cols.append('Behavior Group')

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Cluster',
        color_discrete_sequence=COLOR_PALETTE,
        hover_data=hover_cols if hover_cols else None,
        title='<b>2D PCA Cluster Scatter Plot</b>'
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8, line=dict(width=0.5, color='rgba(255,255,255,0.2)')))
    fig.update_layout(
        xaxis=dict(title='Principal Component 1 (PC1)'),
        yaxis=dict(title='Principal Component 2 (PC2)'),
        template=DARK_TEMPLATE,
        height=550,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_pca_comparison(pca_df, cluster_labels, target_labels):
    """
    Side-by-side or comparison PCA scatter plot: K-Means Clusters vs Actual Activity Labels.
    """
    df_plot = pca_df.copy()
    df_plot['K-Means Cluster'] = [f'Cluster {c}' for c in cluster_labels]
    df_plot['Actual Activity'] = target_labels.values if isinstance(target_labels, pd.Series) else target_labels

    fig = px.scatter(
        df_plot,
        x='PC1',
        y='PC2',
        color='Actual Activity',
        color_discrete_sequence=COLOR_PALETTE,
        title='<b>PCA Visualization by Ground-Truth Activity Label</b>'
    )

    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(
        xaxis=dict(title='Principal Component 1 (PC1)'),
        yaxis=dict(title='Principal Component 2 (PC2)'),
        template=DARK_TEMPLATE,
        height=550,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_activity_distribution(df, target_col):
    """
    Generate bar chart for ground truth activity distribution.
    """
    counts = df[target_col].value_counts().reset_index()
    counts.columns = [target_col, 'Count']
    counts['Percentage'] = (counts['Count'] / counts['Count'].sum() * 100).round(2)

    fig = px.bar(
        counts,
        x=target_col,
        y='Count',
        color=target_col,
        text=counts['Percentage'].apply(lambda x: f"{x}%"),
        color_discrete_sequence=COLOR_PALETTE,
        title=f'<b>Activity Label Frequency Distribution ({target_col})</b>'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(title='Activity Label'),
        yaxis=dict(title='Record Count'),
        template=DARK_TEMPLATE,
        height=450,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_feature_distribution(df, feature_col):
    """
    Generate interactive histogram for a selected sensor feature.
    """
    fig = px.histogram(
        df,
        x=feature_col,
        nbins=50,
        color_discrete_sequence=['#00F2FE'],
        marginal='box',
        title=f'<b>Sensor Feature Distribution: {feature_col}</b>'
    )

    fig.update_layout(
        xaxis=dict(title=feature_col),
        yaxis=dict(title='Frequency'),
        template=DARK_TEMPLATE,
        height=450,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_correlation_heatmap(df, feature_cols, max_features=15):
    """
    Generate Plotly correlation heatmap for a subset of numerical features.
    """
    selected_cols = feature_cols[:max_features]
    corr_matrix = df[selected_cols].corr().round(2)

    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect='auto',
        color_continuous_scale='Viridis',
        title=f'<b>Sensor Feature Correlation Heatmap (Top {len(selected_cols)} Features)</b>'
    )

    fig.update_layout(
        template=DARK_TEMPLATE,
        height=550,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_cluster_distribution(profile_df):
    """
    Bar chart showing total records per identified behavior group.
    """
    fig = px.bar(
        profile_df,
        x='Behavior Group',
        y='Records',
        color='Behavior Group',
        text='Records',
        color_discrete_sequence=COLOR_PALETTE,
        title='<b>Record Count per Activity Behavior Group</b>'
    )

    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis=dict(title='Activity Behavior Group'),
        yaxis=dict(title='Number of Records'),
        template=DARK_TEMPLATE,
        height=450,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig


def plot_feature_comparison(df, cluster_col, feature_cols, profile_df=None):
    """
    Grouped bar chart comparing mean sensor measurements across clusters.
    """
    sample_cols = feature_cols[:6]
    grouped = df.groupby(cluster_col)[sample_cols].mean().reset_index()
    
    group_col = 'Behavior Group'
    # Map cluster ID to behavior group if available
    if profile_df is not None and 'Behavior Group' in profile_df.columns:
        cluster_map = dict(zip(profile_df['Cluster_ID'], profile_df['Behavior Group']))
        grouped[group_col] = grouped[cluster_col].map(cluster_map)
    else:
        grouped[group_col] = grouped[cluster_col].apply(lambda x: f'Cluster {x}')

    melted = grouped.melt(id_vars=[cluster_col, group_col], value_vars=sample_cols, var_name='Sensor Feature', value_name='Mean Value')

    fig = px.bar(
        melted,
        x='Sensor Feature',
        y='Mean Value',
        color=group_col,
        barmode='group',
        color_discrete_sequence=COLOR_PALETTE,
        title='<b>Average Sensor Feature Measurements Across Activity Behavior Groups</b>'
    )

    fig.update_layout(
        xaxis=dict(title='Sensor Measurement Feature'),
        yaxis=dict(title='Mean Normalized Value'),
        template=DARK_TEMPLATE,
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    return fig

