import pandas as pd
from rake_nltk import Rake
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('display.max_columns', 100)
df = pd.read_csv('movie_metadata.csv')
print(df.head())

print(df.shape)
list(df.columns.values)

df = df[['director_name', 'actor_1_name', 'actor_2_name', 'actor_3_name', 'plot_keywords', 'genres', 'movie_title']]

if not df['actor_1_name'].empty or not df['actor_2_name'].empty or not df['actor_3_name'].empty:
    df['actors'] = df['actor_1_name'] + "," + df['actor_2_name'] + "," + df['actor_3_name']

df = df[['director_name', 'plot_keywords', 'genres', 'movie_title', 'actors']]
df.dropna()
print(df.head())

df1 = df.where((pd.notnull(df)), 'REMOVE')
print(df1.head())

df.replace(["NaN"], np.nan, inplace=True)
df = df.dropna()
print(df.head())


for index, row in df.iterrows():
    # process actors names
    app = row['actors'].lower().replace(' ', '')
    app = app.replace(',', ' ')
    row['actors'] = app

    # process director_name
    app = row['director_name'].lower().replace(' ', '')
    row['director_name'] = app

    # process genres
    app = row['genres'].lower().replace('|', ' ')
    row['genres'] = app

    # process plot_keywords
    app = row['plot_keywords'].lower().replace('|', ' ')
    row['plot_keywords'] = app

print(df.head())

df.set_index('movie_title', inplace=True)
print(df.head())

df['bag_of_words'] = ''
columns = df.columns
for index, row in df.iterrows():
    words = ''
    for col in columns:
        words = words + row[col] + ' '
    row['bag_of_words'] = words

df.drop(columns=[col for col in df.columns if col != 'bag_of_words'], inplace=True)

print(df.head())

# instantiating and generating the count matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['bag_of_words'])

# creating a Series for the movie titles so they are associated to an ordered numerical
# list I will use later to match the indexes
indices = pd.Series(df.index)
print(indices[:5])

# generating the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)
print(cosine_sim)


# function that takes in movie title as input and returns the top 10 recommended movies
def recommendations(title, cosine_sim=cosine_sim):
    recommended_movies = []
    idx = -1
    # gettin the index of the movie that matches the title
    for i in range(0, indices.size):
        if indices[i] == title:
            idx = i
            break

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)

    # getting the indexes of the 10 most similar movies
    top_10_indexes = list(score_series.iloc[1:11].index)

    # populating the list with the titles of the best 10 matching movies
    for i in top_10_indexes:
        recommended_movies.append(list(df.index)[i])

    return recommended_movies


print(recommendations('Batman v Superman: Dawn of Justice '))
print(recommendations('Avatar'))