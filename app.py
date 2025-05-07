import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Streamlit Setup
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")
st.title("📊 Customer Segmentation using RFM Analysis + KMeans")

# Upload CSV
uploaded_file = st.file_uploader("Upload your customer transaction CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])

    # Basic Preprocessing
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df.dropna(subset=['CustomerID'], inplace=True)
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

    # RFM Calculation
    snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # RFM Scoring with error handling
    def safe_qcut(series, q, label_order):
        try:
            return pd.qcut(series, q=q, labels=label_order[:q], duplicates='drop')
        except ValueError:
            n_unique = series.nunique()
            adjusted_q = min(q, n_unique)
            return pd.qcut(series, q=adjusted_q, labels=label_order[:adjusted_q], duplicates='drop')

    rfm['R'] = safe_qcut(rfm['Recency'], 5, [5,4,3,2,1])
    rfm['F'] = safe_qcut(rfm['Frequency'], 5, [1,2,3,4,5])
    rfm['M'] = safe_qcut(rfm['Monetary'], 5, [1,2,3,4,5])
    rfm['RFM_Score'] = rfm[['R','F','M']].astype(str).fillna('0').agg(''.join, axis=1)

    # Normalize
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Elbow Plot
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(rfm_scaled)
        sse.append(kmeans.inertia_)

    st.subheader("Elbow Method to Choose Optimal Clusters")
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1,11), sse, marker='o')
    ax1.set_xlabel("Number of Clusters")
    ax1.set_ylabel("SSE")
    ax1.set_title("Elbow Curve")
    st.pyplot(fig1)

    # Cluster Selection
    cluster_num = st.slider("Select number of clusters", 2, 10, 4)
    kmeans = KMeans(n_clusters=cluster_num, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(rfm_scaled)
    rfm['PCA1'] = pca_components[:, 0]
    rfm['PCA2'] = pca_components[:, 1]

    st.subheader("Customer Segments (PCA View)")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=rfm, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax2)
    st.pyplot(fig2)

    # Cluster Summary
    st.subheader("📈 Cluster Summary")
    summary = rfm.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': ['mean', 'count']
    }).round(2)
    st.dataframe(summary)

    # Revenue Pie Chart
    st.subheader("💰 Revenue Contribution by Cluster")
    fig3, ax3 = plt.subplots()
    segment_revenue = rfm.groupby('Cluster')['Monetary'].sum().reset_index()
    ax3.pie(segment_revenue['Monetary'], labels=segment_revenue['Cluster'], autopct='%1.1f%%')
    ax3.axis('equal')
    st.pyplot(fig3)

    # RFM Metric Boxplots
    st.subheader("📊 RFM Metric Distribution by Cluster")
    fig4, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, metric in enumerate(['Recency', 'Frequency', 'Monetary']):
        sns.boxplot(x='Cluster', y=metric, data=rfm, ax=axes[i])
        axes[i].set_title(f'{metric} by Cluster')
    st.pyplot(fig4)

    # Country-level breakdown (optional)
    if 'Country' in df.columns:
        rfm = rfm.merge(df[['CustomerID', 'Country']].drop_duplicates(), on='CustomerID', how='left')
        st.subheader("🌍 Top Countries by Cluster")
        fig5, ax5 = plt.subplots(figsize=(10,6))
        sns.countplot(data=rfm, y='Country', hue='Cluster',
                      order=rfm['Country'].value_counts().index[:10], ax=ax5)
        st.pyplot(fig5)

else:
    st.info("👈 Upload a CSV file to get started.")
