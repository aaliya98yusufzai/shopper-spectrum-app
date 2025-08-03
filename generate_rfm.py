import pandas as pd
import os
from sklearn.cluster import KMeans

# Load original dataset
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# Drop nulls and filter only UK customers
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Country'] == 'United Kingdom']

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create 'TotalAmount' column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Latest date for Recency calculation
latest_date = df['InvoiceDate'].max()

# RFM calculation
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

# Save final RFM data
os.makedirs('data', exist_ok=True)
rfm.to_csv('data/rfm_final.csv', index=False)
print("✅ RFM data saved to data/rfm_final.csv")

# ============================
# Add Clustering to RFM
# ============================

rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]

# Apply KMeans clustering (4 clusters)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_features)

# Save customer_clusters.csv
rfm_clustered = rfm[['Recency', 'Frequency', 'Monetary', 'Cluster']]
rfm_clustered.to_csv("data/customer_clusters.csv", index=False)
print("✅ customer_clusters.csv generated and saved to /data/")
