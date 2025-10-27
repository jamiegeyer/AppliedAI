import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel


# Load MovieLens data (assuming 'movies.csv' with 'movieId', 'title', 'genres')
print("reading movie data...")
movies_df = pd.read_csv('movies.csv')


print("preprocessing  movie data...")

# Preprocess genres: Convert '|' separated genres into a space-separated string
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))

print("vectorizing genres...")
# Create TF-IDF vectorizer for genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])



print("calulating similarities...")
# Calculate cosine similarity between movies
#slow
#cosine_sim = cosine_similarity(genre_matrix, genre_matrix)



def get_recommendations(movie_title, df, top_n=10):
    # Find movie index
    idx = df[df['title'] == movie_title].index[0]
    
    # Compute similarity of that movie only vs all others
    sim_scores = linear_kernel(genre_matrix[idx], genre_matrix).flatten()
    
    # Sort and pick top N
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    
    return df['title'].iloc[sim_indices]


# Function to get recommendations
def get_recommendationsold(movie_title, cosine_sim_matrix, df, top_n=10):
   
    print("in function: gettng indexes (1)...")
    # Get the index of the movie that matches the title
    idx = df[df['title'] == movie_title].index[0]

    print("in function: getting parwise similarity scores (2)...")
    # Get the pairwise similarity scores for that movie
    sim_scores = list(enumerate(cosine_sim_matrix[idx]))
 
    print("in function: sorting movies (3)...")
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top N most similar movies (excluding itself)
    print("in function: finding similar movies (4)...")
    sim_scores = sim_scores[1:top_n+1]

    # Get the movie indices
    print("in function: gettng indexes (5)...")
    movie_indices = [i[0] for i in sim_scores]

    # Return the top N similar movies
    print("in function: returning similar movies (6)...")
    return df['title'].iloc[movie_indices]

# Example usage:
print("running engine...")
recommendations = get_recommendations('Toy Story (1995)', cosine_sim, movies_df)
print(recommendations)
