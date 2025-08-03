import os
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# === Generate missing files if needed ===
# CSV: data/item_similarity.csv
if not os.path.exists('data/item_similarity.csv'):
    from generate_item_similarity import generate_item_similarity
    generate_item_similarity()

# Pickle: models/item_similarity.pkl
if not os.path.exists('models/item_similarity.pkl'):
    from convert_to_pickle import convert_to_pickle
    convert_to_pickle()

# === Load data ===
user_item_matrix = pd.read_csv('data/user_item_matrix.csv', index_col=0)
item_similarity = pd.read_csv('data/item_similarity.csv', index_col=0)

# === Streamlit UI ===
st.set_page_config(page_title="Shopper Spectrum", layout="centered")
st.title("🛍️ Shopper Spectrum: E-Commerce Recommendation App")
st.write("Welcome! Get personalized product recommendations based on customer behavior.")

# === Recommendation Engine ===
st.header("🔍 Get Recommendations")

user_id = st.text_input("Enter User ID (e.g., 17850):")
top_n = st.slider("Select number of recommendations", 1, 20, 5)

def recommend_items(user_id, user_item_matrix, item_similarity, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    user_ratings = user_item_matrix.loc[user_id]
    similar_scores = item_similarity.dot(user_ratings).div(item_similarity.sum(axis=1))

    similar_scores = similar_scores.drop(user_ratings[user_ratings > 0].index, errors='ignore')
    return similar_scores.sort_values(ascending=False).head(top_n).index.tolist()

if st.button("Recommend"):
    if user_id.strip() == "":
        st.warning("Please enter a valid User ID.")
    else:
        try:
            recommendations = recommend_items(user_id, user_item_matrix, item_similarity, top_n)
            if recommendations:
                st.success("Recommended Items:")
                for i, item in enumerate(recommendations, 1):
                    st.write(f"{i}. {item}")
            else:
                st.warning("No recommendations found. This user may not have enough history.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit | Shopper Spectrum Project")
