import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import ast
import seaborn as sns

def generate_visuals():
    print("Generating visualizations for the report...")
    try:
        df = pd.read_csv('tmdb_5000_movies.csv')
    except:
        print("Error: tmdb_5000_movies.csv not found.")
        return

    # --- CHART 1: Word Cloud of Genres ---
    # We extract genres from the JSON format
    def convert(obj):
        L = []
        for i in ast.literal_eval(obj):
            L.append(i['name'])
        return " ".join(L)

    df['genre_list'] = df['genres'].apply(convert)
    all_genres = " ".join(df['genre_list'])

    wordcloud = WordCloud(width=800, height=400, background_color='white', 
                          colormap='viridis').generate(all_genres)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Genres in TMDB Dataset', fontsize=16)
    plt.savefig('genre_wordcloud.png')
    print("- Saved: genre_wordcloud.png")

    # --- CHART 2: Distribution of Ratings ---
    plt.figure(figsize=(10, 6))
    sns.histplot(df['vote_average'], bins=20, kde=True, color='royalblue')
    plt.title('Distribution of Movie Ratings (User Votes)', fontsize=16)
    plt.xlabel('Average Rating (1-10)')
    plt.ylabel('Number of Movies')
    plt.savefig('rating_distribution.png')
    print("- Saved: rating_distribution.png")

if __name__ == "__main__":
    generate_visuals()