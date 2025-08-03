import pandas as pd
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv('online_retail.csv', encoding='ISO-8859-1')

# Drop nulls and filter only UK customers
df.dropna(subset=['CustomerID'], inplace=True)
df = df[df['Country'] == 'United Kingdom']

# Create 'TotalAmount' column
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# Pivot to create customer-item matrix
basket = df.pivot_table(index='CustomerID', columns='Description', values='TotalAmount', aggfunc='sum', fill_value=0)

# Transpose to get item-item matrix
item_user_matrix = basket.T

# Compute cosine similarity between items
item_similarity = cosine_similarity(item_user_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=item_user_matrix.index, columns=item_user_matrix.index)

# Save the item similarity matrix
os.makedirs('data', exist_ok=True)
item_similarity_df.to_csv('data/item_similarity.csv')

print("✅ item_similarity.csv generated and saved to /data/")
