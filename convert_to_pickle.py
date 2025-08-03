import pandas as pd
import pickle

# Load the item similarity CSV
item_similarity = pd.read_csv("data/item_similarity.csv", index_col=0)

# Save as .pkl in models folder
with open("models/item_similarity.pkl", "wb") as f:
    pickle.dump(item_similarity, f)

print("✅ item_similarity.pkl saved in models/")
