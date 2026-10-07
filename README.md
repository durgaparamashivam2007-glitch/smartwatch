# Smartwatch Activity Analysis – Identify Activity Behavior Groups

An end-to-end unsupervised machine learning web application that analyzes wearable smartwatch sensor data to identify and profile distinct activity behavior groups using **K-Means Clustering**, **Principal Component Analysis (PCA)**, and interactive **Streamlit** dashboards.

---

## 🎯 Project Objective
The goal of this project is to analyze high-dimensional wearable sensor measurements (accelerometer, gyroscope, kinetic signals) and automatically group user physical behaviors without supervision. Ground-truth activity labels are retained solely for post-hoc validation and cluster characterization.

---

## ❓ Problem Statement
> *"How can wearable sensor data be analyzed using machine learning to identify meaningful activity behavior groups?"*

---

## 📊 Dataset Information
- **Dataset Source**: Kaggle Activity Recognition Using Wearables / Human Activity Recognition (UCI HAR).
- **Records**: 10,299 sensor readings from smartwatch/smartphone waist-mounted sensors.
- **Features**: 561 multi-axial sensor signals (time and frequency domain signals from accelerometer and gyroscope).
- **Target Label (For evaluation only)**: `Activity` (`WALKING`, `WALKING_UPSTAIRS`, `WALKING_DOWNSTAIRS`, `SITTING`, `STANDING`, `LAYING`).

---

## 🛠️ Technology Stack
- **Programming Language**: Python 3.10+
- **Data Manipulation**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn (StandardScaler, PCA, KMeans)
- **Data Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard Framework**: Streamlit
- **Model Serialization**: Joblib

---

## 🧠 Machine Learning Approach
1. **StandardScaler**: Standardizes sensor features to zero mean and unit variance (\(z = \frac{x - \mu}{\sigma}\)), ensuring equal weighting during distance metric calculations.
2. **PCA (Principal Component Analysis)**: Reduces feature space (561 features) down to 2 principal components for clear 2D spatial cluster visualization while preserving variance.
3. **Elbow Method**: Computes K-Means sum of squared errors (inertia) across \(K \in [2, 8]\) to determine optimal cluster partitioning.
4. **K-Means Clustering**: Partitions sensor feature space into \(K\) activity clusters (`n_init=10`, `random_state=42`).
5. **Data-Driven Cluster Profiling**: Automatically maps clusters to human-readable activity behavior groups (*Highly Active*, *Dynamic Movement*, *Light Activity*, *Sedentary/Stationary*).

---

## 🔄 End-to-End Workflow
```
Dataset (CSV) 
   └──> Data Preprocessing & Cleaning 
         └──> Feature Matrix (X) & Target Isolation
               └──> StandardScaler Normalization 
                     └──> 2D PCA Dimensionality Reduction 
                           └──> Elbow Method (Inertia Analysis)
                                 └──> K-Means Clustering Training
                                       └──> Behavior Group Mapping & Profiling
                                             └──> Streamlit Dashboard & Real-Time Predictor
```

---

## 📁 Project Directory Structure
```
smartwatch-activity-analysis/
│
├── app.py                      # Main Streamlit web application dashboard
├── requirements.txt            # Python dependency requirements
├── README.md                   # Project documentation & deployment guide
├── .gitignore                  # Git ignore rules
│
├── data/
│   └── dataset.csv             # Cleaned smartwatch sensor dataset (10,299 records)
│
├── models/
│   ├── kmeans_model.pkl        # Serialized K-Means clustering model
│   ├── scaler.pkl              # Fitted StandardScaler instance
│   ├── pca_model.pkl           # Fitted PCA model instance
│   ├── feature_names.pkl       # Feature matrix column names list
│   └── cluster_profiles.pkl    # Generated cluster profiling summary
│
├── src/
│   ├── preprocessing.py        # Data loading, cleaning, & auto column detection
│   ├── clustering.py           # Elbow method, PCA reduction, K-Means training & serialization
│   ├── analysis.py             # Cluster profiling & data-driven behavior group mapping
│   └── visualization.py        # Interactive Plotly figures & heatmap generators
│
└── notebooks/
    └── smartwatch_activity_analysis.ipynb # Walkthrough Jupyter Notebook
```

---

## 💻 How to Run Locally

### 1. Clone or Download Repository
```bash
git clone https://github.com/your-username/smartwatch-activity-analysis.git
cd smartwatch-activity-analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Web Application
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 🚀 Deployment Guide (Streamlit Community Cloud)

1. Push your repository to **GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Smartwatch Activity Analysis application"
   git branch -M main
   git remote add origin https://github.com/your-username/smartwatch-activity-analysis.git
   git push -u origin main
   ```

2. Log in to [Streamlit Community Cloud](https://streamlit.io/cloud).
3. Click **New app** and select your GitHub repository (`smartwatch-activity-analysis`).
4. Set **Main file path** to `app.py`.
5. Click **Deploy!**

---

## 🎓 Academic / Demonstration Suitability
This project meets all standards for college Machine Learning course demonstrations, showcasing:
- Strict separation of unsupervised clustering from target labels (avoiding data leakage).
- Dynamic automatic column role detection (robust against unexpected column names).
- Persistent model serialization using Joblib.
- Interactive user prediction interface with dynamic scaling and feature mapping.
