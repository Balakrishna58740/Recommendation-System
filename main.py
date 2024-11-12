import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Sample data of user ratings for movies
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 4, 4],
    'movie_title': [
        'Star Wars', 'The Matrix', 'Inception', 
        'The Matrix', 'Inception', 
        'Star Wars', 'The Matrix', 
        'Inception', 'The Matrix'
    ],
    'rating': [5, 3, 4, 5, 3, 2, 5, 4, 4]
}
df = pd.DataFrame(data)

# Create a matrix of users, items, and ratings
ratings_matrix = df.pivot_table(index='user_id', columns='movie_title', values='rating').fillna(0)

# Compute cosine similarity between items
item_similarity = cosine_similarity(ratings_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity, index=ratings_matrix.columns, columns=ratings_matrix.columns)

# Function to recommend movies based on item similarity
def recommend_movies(movie_title, similarity_df, num_recommendations=3):
    # Find similar movies based on similarity score
    similar_movies = similarity_df[movie_title].sort_values(ascending=False)
    return similar_movies[1:num_recommendations + 1]

# Example usage
print("Recommendations for 'Star Wars':")
print(recommend_movies('Star wars', item_similarity_df))
