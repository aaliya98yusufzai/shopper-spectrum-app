import pandas as pd

# Load the dataset
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')

# Drop rows with missing CustomerID or Description
df.dropna(subset=['CustomerID', 'Description'], inplace=True)

# Strip whitespace and remove empty descriptions
df['Description'] = df['Description'].str.strip()
df = df[df['Description'] != '']

# Create user-item matrix: rows = CustomerID, columns = Items, values = purchase count
user_item_matrix = df.groupby(['CustomerID', 'Description'])['Quantity'].sum().unstack(fill_value=0)

# Convert CustomerID to string to match with Streamlit logic
user_item_matrix.index = user_item_matrix.index.astype(str)

# Save to CSV
user_item_matrix.to_csv("data/user_item_matrix.csv")

print("✅ user_item_matrix.csv saved to data/")
