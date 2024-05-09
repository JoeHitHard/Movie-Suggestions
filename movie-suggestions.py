import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

df = pd.read_csv('./netflix_titles.csv', encoding='iso-8859-1')

df.fillna('', inplace=True)

df['combined_features'] = df['type'] + ' ' + df['director'] + ' ' + df['cast'] + ' ' + df['country'] + ' ' + df[
    'rating'] + ' ' + df['duration'] + ' ' + df['listed_in'] + ' ' + df['description']

tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(df['combined_features'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


def recommend(title, cosine_sim=cosine_sim):
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the title itself
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]


recommendations = recommend('Little Things')
print(recommendations)

import plotly.graph_objects as go

# Get the cosine similarity scores for the first 10 movies
number_of_movies = 100
scores = cosine_sim[:number_of_movies, :number_of_movies]

score_data = []
for i in range(number_of_movies):
    scores[i][i] = 0
    score_data.append(
        go.Heatmap(z=scores, x=list(df['title'][:number_of_movies]), y=list(df['title'][:number_of_movies]),
                   name=df['title'][i], colorscale='Greys'))

# Create a bar chart
fig = go.Figure(data=score_data)

# Update layout
fig.update_layout(title='Cosine Similarity Scores of First ' + str(number_of_movies) + ' Movies', xaxis_title='Movie Title',
                  yaxis_title='Cosine Similarity Score')
# Convert the plot to HTML
graph_html = fig.to_html()

with open('cosine_similarity_' + str(number_of_movies) + '.html', 'w') as f:
    f.write(graph_html)
