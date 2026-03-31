# 🎬 Movie Intelligence: Metadata-Based Recommendation System

## 📌 Project Overview
This project is an advanced **Content-Based Recommendation Engine** built using the TMDB 5000 dataset. Unlike simple recommendation scripts that only look at one feature (like genre), this system builds a **"Metadata Profile"** for every movie. It processes over 5,000 films to find deep structural and creative similarities by analyzing directors, lead actors, plot keywords, and genres simultaneously.

The core of this project lies in **Natural Language Processing (NLP)**—converting human-readable descriptions into mathematical vectors to calculate the "Cosine Distance" between different pieces of art.

---

## 🧠 The AI Logic & Workflow

### 1. Data Merging & Relational Mapping
The system begins by merging two separate data sources: `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv`. This allows the AI to "know" not just what a movie is about (overview), but who created it (crew) and who starred in it (cast).

### 2. Feature Engineering: The "Metadata Soup"
A critical step in this project is the creation of a **Metadata Soup**. We don't just look at words; we clean them to ensure high-quality matching:
* **Space Elimination:** Names like "Johnny Depp" are converted to "JohnnyDepp". This prevents the AI from confusing "Johnny Depp" with "Johnny Knoxville" based solely on the first name "Johnny."
* **Keyword Extraction:** We use `ast.literal_eval` to parse complex JSON strings into Python lists, extracting only the most relevant tags.
* **Vectorization:** We use `CountVectorizer` to transform these text tags into a 5,000-dimensional coordinate system.

### 3. Mathematical Model: Cosine Similarity
To determine how similar "Movie A" is to "Movie B," the system calculates the **Cosine Angle** between their vectors. 



* **Formula:** $similarity = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$
* Instead of measuring the "distance" (which can be skewed by the length of the movie description), we measure the **direction** of the vector. If two movies point in the same direction in our 5,000-dimensional space, they are recommended.

---

## 🛠️ Technical Stack
* **Language:** Python 3.10+
* **Data Manipulation:** `Pandas`, `NumPy`
* **Machine Learning:** `Scikit-Learn` (CountVectorizer, Cosine Similarity)
* **Visualizations:** `Matplotlib`, `Seaborn`, `WordCloud`
* **Data Cleaning:** `ast` (Abstract Syntax Trees)

---

## 🚀 Installation and Execution

### 1. Prerequisites
Ensure you have the following libraries installed via your terminal:
```bash
pip install pandas scikit-learn matplotlib seaborn wordcloud
