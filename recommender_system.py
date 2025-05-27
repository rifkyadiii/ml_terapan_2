#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[274]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Data Loading

# In[275]:


# Membaca data ratings
ratings_cols = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('dataset/u.data', sep='\t', names=ratings_cols, encoding='latin-1')


# In[276]:


ratings.head()


# In[277]:


# Membaca data movies  
movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
          'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
          'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies_cols.extend(genres)
movies = pd.read_csv('dataset/u.item', sep='|', names=movies_cols, encoding='latin-1')


# In[278]:


movies.head()


# ## Data Understanding

# In[279]:



# ## Modeling

# Content-Based Filtering

# In[294]:


# TF-IDF Vectorization untuk genre
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_genres['genre_string'])


# In[295]:


# Menghitung cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# In[296]:


# Membuat mapping judul ke index
indices = pd.Series(movies_with_genres.index, index=movies_with_genres['title']).drop_duplicates()


# In[297]:


# Fungsi rekomendasi content-based
def get_content_recommendations(title, cosine_sim=cosine_sim, df=movies_with_genres, top_n=10):
    """
    Memberikan rekomendasi berdasarkan content similarity
    """
    try:
        # Pastikan judul ada di index
        if title not in indices:
            return f"Film '{title}' tidak ditemukan dalam dataset."

        # Ambil index film
        idx = indices[title]

        # Ambil vektor similarity untuk film tersebut (pastikan 1D array)
        sim_scores = list(enumerate(cosine_sim[idx]))

        # Urutkan berdasarkan similarity score (descending)
        sim_scores = sorted(sim_scores, key=lambda x: float(x[1]), reverse=True)

        # Ambil top N (kecuali diri sendiri di posisi pertama)
        sim_scores = sim_scores[1:top_n + 1]

        # Ambil index film yang disarankan
        movie_indices = [i[0] for i in sim_scores]

        # Ambil data film
        recommendations = df.iloc[movie_indices][['title', 'genre_string']].copy()
        recommendations['similarity_score'] = [float(score[1]) for score in sim_scores]

        return recommendations

    except Exception as e:
        return f"Terjadi error saat mencari rekomendasi untuk '{title}': {str(e)}"


# In[298]:


# Test rekomendasi content-based
test_movie = "Toy Story (1995)"
print(f"\nRekomendasi Content-Based untuk '{test_movie}':")
content_recs = get_content_recommendations(test_movie, top_n=5)
if isinstance(content_recs, pd.DataFrame):
    for i, (_, row) in enumerate(content_recs.iterrows(), 1):
        print(f"{i}. {row['title']}")
        print(f"   Genre: {row['genre_string']}")
        print(f"   Similarity: {row['similarity_score']:.3f}")
        print()
else:
    print(content_recs)


# Collaborative Filtering

# In[299]:


# Menggunakan SVD untuk matrix factorization
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_item_matrix)
item_factors = svd.components_


# In[300]:


# Rekonstruksi matrix
predicted_ratings = np.dot(user_factors, item_factors)
predicted_ratings_df = pd.DataFrame(predicted_ratings, 
                                  index=user_item_matrix.index, 
                                  columns=user_item_matrix.columns)


# In[301]:


# Fungsi rekomendasi collaborative filtering
def get_collaborative_recommendations(user_id, predicted_ratings_df=predicted_ratings_df,
                                    train_user_item_matrix=train_user_item_matrix,
                                    movies_df=movies, top_n=10):
    
    if user_id not in predicted_ratings_df.index:
        return f"User ID {user_id} tidak ditemukan dalam model training."

    user_predictions = predicted_ratings_df.loc[user_id]

    # Ambil film yang belum dirating OLEH USER DI DATA TRAINING
    all_movie_ids = movies_df['movie_id'].unique()
    rated_movies_in_train = train_user_item_matrix.loc[user_id][train_user_item_matrix.loc[user_id] > 0].index.tolist()
    unrated_movies_ids = [mid for mid in all_movie_ids if mid not in rated_movies_in_train]

    # Filter prediksi untuk film yang belum dirating
    recommendations_for_unrated = user_predictions.reindex(unrated_movies_ids, fill_value=0)

    # Sort berdasarkan predicted rating
    recommendations_for_unrated = recommendations_for_unrated.sort_values(ascending=False).head(top_n)

    # Merge dengan informasi film
    rec_with_titles = []
    for movie_id, pred_rating in recommendations_for_unrated.items():
        movie_info = movies_df[movies_df['movie_id'] == movie_id]
        if not movie_info.empty:
            rec_with_titles.append({
                'movie_id': movie_id,
                'title': movie_info.iloc[0]['title'],
                'predicted_rating': pred_rating
            })

    return pd.DataFrame(rec_with_titles)


# In[302]:


# Test rekomendasi collaborative filtering
test_user = 1
print(f"\nRekomendasi Collaborative Filtering untuk User {test_user}:")
collab_recs = get_collaborative_recommendations(test_user, top_n=5)
if isinstance(collab_recs, pd.DataFrame):
    for i, (_, row) in enumerate(collab_recs.iterrows(), 1):
        print(f"{i}. {row['title']}")
        print(f"   Predicted Rating: {row['predicted_rating']:.2f}")
        print()
else:
    print(collab_recs)


# ## Evaluation

# In[303]:


# Evaluasi Content-Based FilterinG
def evaluate_content_based(sample_titles, top_n=10):
    similarity_scores = []

    for title in sample_titles:
        recs = get_content_recommendations(title, top_n=top_n)
        if isinstance(recs, pd.DataFrame):
            similarity_scores.extend(recs['similarity_score'].values)

    if similarity_scores:
        mean_score = np.mean(similarity_scores)
        coverage = (len(similarity_scores) / (len(sample_titles) * top_n)) * 100
        return round(mean_score, 4), round(coverage, 1)
    else:
        return None, 0.0


# In[304]:


# Evaluasi Collaborative Filtering - RMSE & MAE
def evaluate_collaborative_rmse(test_user_item_matrix, predicted_ratings_df):
    actuals, predictions = [], []

    for user in test_user_item_matrix.index:
        for item in test_user_item_matrix.columns:
            actual_rating = test_user_item_matrix.loc[user, item]
            if actual_rating > 0:
                actuals.append(actual_rating)
                predictions.append(predicted_ratings_df.loc[user, item])

    if actuals:
        rmse = sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        return round(rmse, 4), round(mae, 4)
    else:
        return None, None


# In[305]:


# Evaluasi Collaborative Filtering - Precision@K
def precision_at_k(user_id, predicted_ratings_df, test_user_item_matrix, k=10, threshold=4.0):
    if user_id not in predicted_ratings_df.index:
        return None

    user_preds = predicted_ratings_df.loc[user_id]
    user_actuals = test_user_item_matrix.loc[user_id]

    relevant_items = user_actuals[user_actuals >= threshold].index.tolist()
    test_items = user_actuals[user_actuals > 0].index.tolist()

    if not test_items:
        return None

    top_k_items = user_preds[test_items].sort_values(ascending=False).head(k).index.tolist()
    num_relevant = len(set(top_k_items) & set(relevant_items))

    return num_relevant / k

def evaluate_precision_at_k(predicted_ratings_df, test_user_item_matrix, k=5, threshold=4.0):
    precisions = []

    for user_id in test_user_item_matrix.index:
        p_at_k = precision_at_k(user_id, predicted_ratings_df, test_user_item_matrix, k=k, threshold=threshold)
        if p_at_k is not None:
            precisions.append(p_at_k)

    return round(np.mean(precisions), 4) if precisions else None


# In[306]:


sample_movies = movies_with_genres['title'].sample(10, random_state=42).tolist()

# Content-Based
content_mean_score, coverage_content = evaluate_content_based(sample_movies, top_n=5)

# Collaborative Filtering
rmse_score, mae_score = evaluate_collaborative_rmse(test_user_item_matrix, predicted_ratings_df)
avg_precision = evaluate_precision_at_k(predicted_ratings_df, test_user_item_matrix, k=5, threshold=4.0)

print("\nRingkasan Performa:")
print(f"- RMSE Collaborative       : {rmse_score:.3f}" if rmse_score else "- RMSE: Tidak tersedia")
print(f"- MAE Collaborative        : {mae_score:.3f}" if mae_score else "- MAE: Tidak tersedia")
print(f"- Precision@5              : {avg_precision:.3f}" if avg_precision else "- Precision@5: Tidak tersedia")
print(f"- Content-Based Mean Sim   : {content_mean_score:.3f}" if content_mean_score else "- Similarity: Tidak tersedia")
print(f"- Coverage Content-Based   : {coverage_content:.1f}%")


# ## Inferensi

# In[307]:


# Content-based untuk user berdasarkan film favorit
user_1_ratings = ratings[ratings['user_id'] == 1].sort_values('rating', ascending=False)
if len(user_1_ratings) > 0:
    fav_movie_id = user_1_ratings.iloc[0]['item_id']
    fav_movie_title = movies[movies['movie_id'] == fav_movie_id]['title'].iloc[0]
    
    print(f"\nFilm dengan rating tertinggi dari User 1: {fav_movie_title}")
    
    # Content-based recommendations
    print("\nRekomendasi Content-Based:")
    content_hybrid = get_content_recommendations(fav_movie_title, top_n=3)
    if isinstance(content_hybrid, pd.DataFrame):
        for i, (_, row) in enumerate(content_hybrid.iterrows(), 1):
            print(f"{i}. {row['title']} (Sim: {row['similarity_score']:.3f})")

# Collaborative recommendations
print("\nRekomendasi Collaborative:")
collab_hybrid = get_collaborative_recommendations(1, top_n=3)
if isinstance(collab_hybrid, pd.DataFrame):
    for i, (_, row) in enumerate(collab_hybrid.iterrows(), 1):
        print(f"{i}. {row['title']} (Pred: {row['predicted_rating']:.2f})")


# In[308]:


# Menyimpan hasil untuk laporan
evaluation_results = {
    'rmse': rmse_score if rmse_score else 'N/A',
    'precision_at_5': avg_precision,
    'coverage_content': coverage_content,
    'total_movies': len(movies),
    'total_ratings': len(ratings),
    'total_users': ratings['user_id'].nunique()
}

print(f"\nHasil Evaluasi Final:")
for key, value in evaluation_results.items():
    print(f"{key}: {value}")

