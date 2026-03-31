import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_data():
    # 1. Load both files
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')

    # 2. Merge them into one big dataset
    movies = movies.merge(credits, on='title')

    # 3. Select only the columns we need
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    # 4. Helper function to extract names from JSON strings
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return L

    # 5. Specialized function to get just the Director's name
    def fetch_director(obj):
        L = []
        for i in ast.literal_eval(obj):
            if i['job'] == 'Director':
                L.append(i['name'])
                break
        return L

    # Apply cleaning to all columns
    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3]) # Just top 3 actors
    movies['crew'] = movies['crew'].apply(fetch_director)
    
    # Remove spaces so "Johnny Depp" becomes "JohnnyDepp" (crucial for AI)
    def collapse(L):
        L1 = []
        for i in L:
            L1.append(i.replace(" ",""))
        return L1

    movies['cast'] = movies['cast'].apply(collapse)
    movies['crew'] = movies['crew'].apply(collapse)
    movies['genres'] = movies['genres'].apply(collapse)
    movies['keywords'] = movies['keywords'].apply(collapse)

    # Create the "Soup" (Tags)
    movies['overview'] = movies['overview'].apply(lambda x:x.split())
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    
    # Convert list back to a single string
    new_df = movies[['movie_id', 'title', 'tags']].copy()
    new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x).lower())
    
    return new_df

def recommend(movie, df, similarity):
    try:
        movie_index = df[df['title'].str.lower() == movie.lower()].index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        
        print(f"\nRecommended for you:")
        for i in movies_list:
            print(f"-> {df.iloc[i[0]].title}")
    except:
        print("Movie not found in the 5,000 film database.")

if __name__ == "__main__":
    print("Processing 5,000 movies... please wait.")
    final_df = clean_data()
    
    # Vectorization
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(final_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    while True:
        user_choice = input("\nEnter a movie title (or 'exit'): ")
        if user_choice.lower() == 'exit': break
        recommend(user_choice, final_df, similarity)
