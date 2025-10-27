import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import time
pd.set_option('display.max_rows', 10)

print("++++ MOVIE RECOMMENDER V0.1 - JAMIE GEYER- APPLIED AI ++++")

time.sleep(5)


# Load MovieLens data (assuming 'movies.csv' with 'movieId', 'title', 'genres')
#print("reading movie data...")
movies_df = pd.read_csv('movies.csv')


#print("preprocessing  movie data...")

# Preprocess genres: Convert '|' separated genres into a space-separated string
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.replace('|', ' '))

randoms=movies_df.sample(n=10)[['movieId', 'title', 'genres']]


print(randoms)
print("++++++++++++++++++++++++++++++++")
option=input("Enter the Movie ID for favorite movie from the list:    ")

print("++++++++++++++++++++++++++++++++")

faves_df=randoms.query("movieId == " + option)
faves_title=faves_df['title']

print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")

print("you selected: " + faves_title)
print("will generate some recommendations for you now...")
time.sleep(5)
faves_title_str=faves_title.values[0]
print(faves_title_str)

#print("vectorizing genres...")
# Create TF-IDF vectorizer for genres
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])


def get_recommendations(movie_title, df, top_n=10):
    # Find movie index
    idx = df[df['title'] == movie_title].index[0]
    
    # Compute similarity of that movie only vs all others
    sim_scores = linear_kernel(genre_matrix[idx], genre_matrix).flatten()
    
    # Sort and pick top N
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    
    return df[['title','genres']].iloc[sim_indices]


print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")
print()
print("Please find below 5 recommendations.  Enjoy!!")


recommendations = get_recommendations(faves_title_str, movies_df, top_n=5)
print(recommendations)

print()
print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")
print("++++++++++++++++++++++++++++++++")
print()
