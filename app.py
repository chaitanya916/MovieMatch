import pandas as pd
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os
import streamlit as st
import numpy as np

# Load the MovieLens dataset
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Preprocess the data
movies['genres'] = movies['genres'].str.replace('|', ' ')
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(movies['genres'])

# Calculate cosine similarity for content-based filtering
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

# User profiles database (for simplicity, using a dictionary)
user_profiles = {}
user_profiles_file = 'user_profiles.json'

# Load user profiles from file if it exists and is valid
if os.path.exists(user_profiles_file):
    try:
        with open(user_profiles_file, 'r') as file:
            user_profiles = json.load(file)
    except json.JSONDecodeError:
        user_profiles = {}  # Initialize as empty if JSON is invalid

# Function to save user profiles to file
def save_user_profiles():
    with open(user_profiles_file, 'w') as file:
        json.dump(user_profiles, file)

# Function to create or update user profile
def update_user_profile(user_id: str, rated_movies: dict) -> None:
    user_id = str(user_id)  # Ensure user_id is a string
    if user_id not in user_profiles:
        user_profiles[user_id] = {'rated_movies': {}, 'genre_counts': {}, 'recommended_movies': []}
    for movie_id, rating in rated_movies.items():
        movie_id = str(movie_id)  # Ensure movie_id is a string
        user_profiles[user_id]['rated_movies'][movie_id] = rating
        genres = movies[movies['movieId'] == int(movie_id)]['genres'].values[0].split()
        for genre in genres:
            if genre not in user_profiles[user_id]['genre_counts']:
                user_profiles[user_id]['genre_counts'][genre] = 0
            user_profiles[user_id]['genre_counts'][genre] += rating
    save_user_profiles()

# Function to get popular movies from selected genres
def get_popular_movies_from_genres(genres: list, top_n: int = 5) -> pd.DataFrame:
    filtered_movies = movies[movies['genres'].apply(lambda x: any(genre in x for genre in genres))]
    movie_ratings = ratings[ratings['movieId'].isin(filtered_movies['movieId'])]
    avg_ratings = movie_ratings.groupby('movieId')['rating'].mean()
    rating_counts = movie_ratings.groupby('movieId')['rating'].count()
    weighted_ratings = (avg_ratings * rating_counts) / (rating_counts + 10)
    popular_movies = weighted_ratings.sort_values(ascending=False).head(top_n)
    return filtered_movies[filtered_movies['movieId'].isin(popular_movies.index)]

# Function to get the top 5 movies for each genre
def get_top_movies_by_genre(top_n: int = 5) -> dict:
    genres = movies['genres'].str.split(expand=True).stack().unique()
    top_movies_by_genre = {}
    for genre in genres:
        top_movies_by_genre[genre] = get_popular_movies_from_genres([genre], top_n=top_n)
    return top_movies_by_genre

# Content-based recommendations
def content_based_recommendations(user_id: str, top_n: int = 10) -> list:
    user_id = str(user_id)  # Ensure user_id is a string
    if user_id not in user_profiles or not user_profiles[user_id]['rated_movies']:
        return []
    user_data = user_profiles[user_id]
    rated_movie_ids = user_data['rated_movies'].keys()
    if not rated_movie_ids:
        return []
    movie_indices = [movies[movies['movieId'] == int(mid)].index[0] for mid in rated_movie_ids]
    sim_scores = cosine_sim[movie_indices].mean(axis=0)
    recommended_indices = sim_scores.argsort()[-(top_n * 2):][::-1]  # Get more than needed to filter out already recommended
    recommended_movies = movies.iloc[recommended_indices]
    recommended_movie_ids = recommended_movies['movieId'].values.tolist()
    # Filter out already recommended movies
    recommended_movie_ids = [mid for mid in recommended_movie_ids if mid not in user_profiles[user_id]['recommended_movies']]
    return recommended_movie_ids[:top_n]

# Collaborative filtering using Surprise (SVD)
def collaborative_filtering():
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.25)
    algo = SVD()
    algo.fit(trainset)
    return algo

algo = collaborative_filtering()

# Compute User-User Similarity Using Pearson Correlation
def compute_pearson_similarity(user_item_matrix):
    user_similarity = user_item_matrix.T.corr(method='pearson')
    return user_similarity

user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')
user_similarity_matrix = compute_pearson_similarity(user_item_matrix)

# Generate Pearson-Based Recommendations
def get_pearson_recommendations(user_id, top_n=10):
    user_id = int(user_id)
    if user_id not in user_item_matrix.index:
        return []

    similarity_scores = user_similarity_matrix[user_id].dropna()
    similarity_scores = similarity_scores[similarity_scores.index != user_id]

    similar_users = similarity_scores.nlargest(top_n).index

    similar_users_ratings = user_item_matrix.loc[similar_users]

    weighted_ratings = similar_users_ratings.T.dot(similarity_scores[similar_users])
    sum_of_weights = similarity_scores[similar_users].sum()

    predicted_ratings = weighted_ratings / sum_of_weights

    user_rated_movies = user_item_matrix.loc[user_id].dropna().index
    predicted_ratings = predicted_ratings.drop(user_rated_movies, errors='ignore')

    recommended_movies = predicted_ratings.nlargest(top_n).index
    return recommended_movies.tolist()

# Function to normalize a list of scores
def normalize(scores):
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [1.0 for _ in scores]  # If all scores are the same, return all 1s
    else:
        return [(score - min_score) / (max_score - min_score) for score in scores]

# Function to get content-based recommendations with scores
def get_content_based_scores(user_id):
    user_id = str(user_id)
    if user_id not in user_profiles or not user_profiles[user_id]['rated_movies']:
        return []
    user_data = user_profiles[user_id]
    rated_movie_ids = user_data['rated_movies'].keys()
    if not rated_movie_ids:
        return []
    movie_indices = [movies[movies['movieId'] == int(mid)].index[0] for mid in rated_movie_ids]
    sim_scores = cosine_sim[movie_indices].mean(axis=0)
    recommended_indices = sim_scores.argsort()[::-1]
    recommended_movies = movies.iloc[recommended_indices]
    recommended_movie_ids = recommended_movies['movieId'].values.tolist()
    scores = sim_scores[recommended_indices].tolist()
    return list(zip(recommended_movie_ids, scores))

# Function to get SVD-based collaborative filtering recommendations with scores
def get_svd_collaborative_scores(user_id):
    user_id = int(user_id)
    user_ratings = ratings[ratings['userId'] == user_id]
    rated_movie_ids = user_ratings['movieId'].tolist()
    user_unrated_movies = ratings[~ratings['movieId'].isin(rated_movie_ids)]
    user_unrated_movies = user_unrated_movies.drop_duplicates(subset='movieId')

    svd_recs = []
    for movie_id in user_unrated_movies['movieId']:
        pred = algo.predict(user_id, movie_id)
        svd_recs.append((movie_id, pred.est))

    svd_recs.sort(key=lambda x: x[1], reverse=True)
    return svd_recs[:10]  # Return top 10 recommendations with scores

# Function to get Pearson collaborative filtering recommendations with scores
def get_pearson_collaborative_scores(user_id):
    user_id = str(user_id)
    pearson_recs = get_pearson_recommendations(user_id)
    pearson_scores = np.ones(len(pearson_recs))  # Dummy scores for demonstration
    return list(zip(pearson_recs, pearson_scores))

# Hybrid recommendation function with weighted scores
def hybrid_recommendations(user_id: str, top_n: int = 10, weight_content: float = 0.5,
                           weight_collaborative: float = 0.5, weight_pearson: float = 0.5) -> list:
    user_id = str(user_id)
    content_recs = get_content_based_scores(user_id)
    svd_recs = get_svd_collaborative_scores(user_id)
    pearson_recs = get_pearson_collaborative_scores(user_id)

    if not content_recs and not svd_recs and not pearson_recs:
        return []

    if content_recs:
        content_movie_ids, content_scores = zip(*content_recs)
        content_scores = normalize(content_scores)
    else:
        content_movie_ids, content_scores = [], []

    if svd_recs:
        svd_movie_ids, svd_scores = zip(*svd_recs)
        svd_scores = normalize(svd_scores)
    else:
        svd_movie_ids, svd_scores = [], []

    if pearson_recs:
        pearson_movie_ids, pearson_scores = zip(*pearson_recs)
        pearson_scores = normalize(pearson_scores)
    else:
        pearson_movie_ids, pearson_scores = [], []

    all_movie_ids = list(set(content_movie_ids) | set(svd_movie_ids) | set(pearson_movie_ids))
    final_scores = {}

    for movie_id in all_movie_ids:
        final_scores[movie_id] = 0
        if movie_id in content_movie_ids:
            final_scores[movie_id] += weight_content * content_scores[content_movie_ids.index(movie_id)]
        if movie_id in svd_movie_ids:
            final_scores[movie_id] += weight_collaborative * svd_scores[svd_movie_ids.index(movie_id)]
        if movie_id in pearson_movie_ids:
            final_scores[movie_id] += weight_pearson * pearson_scores[pearson_movie_ids.index(movie_id)]

    sorted_movie_ids = sorted(final_scores, key=final_scores.get, reverse=True)
    recommended_movie_ids = [mid for mid in sorted_movie_ids if
                             mid not in user_profiles.get(user_id, {}).get('recommended_movies', [])][:top_n]

    if user_id not in user_profiles:
        user_profiles[user_id] = {'rated_movies': {}, 'genre_counts': {}, 'recommended_movies': []}

    user_profiles[user_id]['recommended_movies'].extend(recommended_movie_ids)
    save_user_profiles()

    return recommended_movie_ids

# Streamlit interface for new users
st.title('Movie Recommendation System')

st.header('New User')
new_user_id = st.text_input("Enter a new user ID:", key='new_user_id')
if new_user_id:
    new_user_id = str(new_user_id)
    if new_user_id not in user_profiles:
        st.write("Select the genres you like:")
        genres = movies['genres'].str.split(expand=True).stack().unique()
        selected_genres = st.multiselect("Genres", genres, key='selected_genres')
        if selected_genres:
            if st.button('Get Genre-Based Recommendations', key='get_genre_recommendations'):
                st.write("Here are some recommendations for you:")
                genre_recommendations = get_popular_movies_from_genres(selected_genres, top_n=10)
                for _, row in genre_recommendations.iterrows():
                    st.write(row['title'])
            else:
                st.write("Please rate the following movies:")
                movies_to_rate = get_popular_movies_from_genres(selected_genres, top_n=10)
                initial_ratings = {}
                for index, row in movies_to_rate.iterrows():
                    movie_title = row['title']
                    rating = st.slider(f"Rate {movie_title}", 0.5, 5.0, step=0.5, key=f"rating_{index}")
                    if rating:
                        initial_ratings[row['movieId']] = rating
                    if st.button(f"Skip {movie_title}", key=f"skip_{index}"):
                        continue
                if st.button('Submit Ratings', key='submit_ratings'):
                    update_user_profile(new_user_id, initial_ratings)
                    st.write("Profile created and ratings saved successfully.")
                    st.write("Here are some recommendations for you:")
                    recommendations = hybrid_recommendations(new_user_id)
                    for movie in recommendations:
                        st.write(movies[movies['movieId'] == movie]['title'].values[0])

# Streamlit interface to get recommendations
st.header('Get Recommendations')
user_id_recommend = st.text_input("Enter your user ID to get recommendations:", key='user_id_recommend')
if user_id_recommend:
    user_id_recommend = str(user_id_recommend)
    if user_id_recommend in user_profiles or int(user_id_recommend) in ratings['userId'].unique():
        if st.button('Get Recommendations', key='get_recommendations_' + user_id_recommend):
            recommended_movies = hybrid_recommendations(user_id_recommend)
            st.write("Recommended movies for you:")
            for movie in recommended_movies:
                st.write(movies[movies['movieId'] == movie]['title'].values[0])
    else:
        st.write("User ID not found. Please create a new profile in the 'New User' section.")

# Streamlit interface for rating movies
st.header('Rate Movies')
user_id_rate = st.text_input("Enter your user ID to rate movies:", key='user_id_rate')
if user_id_rate:
    user_id_rate = str(user_id_rate)
    if user_id_rate in user_profiles or int(user_id_rate) in ratings['userId'].unique():
        movie_name = st.selectbox("Enter the name of the movie you want to rate:", movies['title'].unique())
        rating = st.text_input("Enter your rating for the movie (0.5 to 5):")

        if st.button('Rate Movie', key='rate_movie'):
            if movie_name and rating:
                try:
                    rating = float(rating)
                    if 0.5 <= rating <= 5:
                        movie_id = movies[movies['title'] == movie_name].iloc[0]['movieId']
                        update_user_profile(user_id_rate, {str(movie_id): rating})
                        st.write("Rating saved successfully.")
                    else:
                        st.write("Invalid rating. Please enter a value between 0.5 and 5.")
                except ValueError:
                    st.write("Invalid rating. Please enter a valid number.")
            else:
                st.write("Please enter both movie name and rating.")
    else:
        st.write("User ID not found. Please create a new profile in the 'New User' section.")

# Sidebar to display popular movies by genre
def display_popular_movies_sidebar():
    st.sidebar.header('Top 5 Movies by Genre')
    top_movies_by_genre = get_top_movies_by_genre()
    for genre, movies_df in top_movies_by_genre.items():
        st.sidebar.subheader(genre)
        for _, row in movies_df.iterrows():
            st.sidebar.write(row['title'])

# Initial display of popular movies
display_popular_movies_sidebar()
