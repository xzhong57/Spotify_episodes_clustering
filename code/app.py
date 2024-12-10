from faicons import icon_svg
from shiny import App, ui, render, reactive, Inputs, Outputs, Session
from shinywidgets import output_widget, render_widget
import plotly.express as px

# Spotify API
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

client_id = '9bd3a02927954156971b2be285f9dcee'
client_secret = 'c5f057706ed64931942b80729eb4e35c'

client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

# Word count
from collections import Counter
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words += ['show', 'shows', 'podcast', 'podcasts', 'episode', 'episodes']

def preprocess_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    tokens = word_tokenize(text.lower())
    meaningful_words = [word for word in tokens if word.isalnum() and word not in stop_words]
    return meaningful_words

with open("filtered_words.txt", "r") as f:
    filtered_words = [line.strip() for line in f]

# Load data and models
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
import joblib
from scipy.sparse import csr_matrix

shows_data = pd.read_csv('shows_data.csv')
episodes_data = pd.read_csv('episodes_data.csv')

principal_components = np.load('principal_components.npy')
kmeans_labels = np.load('kmeans_labels.npy')

svd = joblib.load('pca_model.pkl')
kmeans = joblib.load('kmeans.pkl')

#------------------------------------------------------------------------------

# Define UI
app_ui = ui.page_fillable(
    ui.tags.div(
        ui.h1("Spotify Podcasts and Episodes", style="text-align: left; margin-top: 0;"),
        style="position: absolute; top: 10px; left: 10px; z-index: 1000;"
    ),
    
    ui.navset_card_pill(
        ui.nav_spacer(),
        ui.nav_panel(
            "Episode Clustering",
            "Find the nearest episodes for the chosen episode",
            ui.page_sidebar(
                ui.sidebar(
                    # User Input Section
                    ui.input_text("show", "Enter a podcast's name"),
                    ui.input_action_button("search_show", "Search", theme="gradient-blue-indigo"),
                ),
            ),
            
            ui.card(
                ui.layout_columns(
                    ui.card(output_widget("pc12")),
                    ui.card(output_widget("pc23"))
                ),
                ui.layout_columns(
                    ui.card(output_widget("pc13")),
                    ui.card(output_widget("pc24"))
                ),
                ui.layout_columns(
                    ui.card(output_widget("pc14")),
                    ui.card(output_widget("pc34"))
                )
            )
        )
    ),
            
    ui.card(
        ui.tags.footer(
            "Data source: https://developer.spotify.com/ | Contact: https://github.com/xzhong57/Spotify_episodes_clustering",
            style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #f8f9fa; z-index: 1000; text-align: center; padding: 10px;"
        )
    )
)

#------------------------------------------------------------------------------

# Define server
def server(input: Inputs, output: Outputs, session: Session):
    shows = reactive.Value(None)
    episodes = reactive.Value(None)
    episode = reactive.Value(None)
    pc = reactive.Value(None)
    label = reactive.Value(None)
    
    #------------------------Episode Clustering--------------------------------
    @reactive.effect
    @reactive.event(input.search_show)
    def _():
        shows.set(sp.search(q=input.show(), limit = 10, type='show')['shows']['items'])
        names = [show['name'] for show in shows.get()]
        
        ui.insert_ui(
            ui.input_select("show_name", "Select a podcast", choices=names),
            selector="#add",
            where="afterEnd",
        )
        
        ui.insert_ui(
            ui.input_action_button("fetch_episode", "Fetch episodes", theme="gradient-blue-indigo"),
            selector="#add",
            where="afterEnd",
        )
    
    @reactive.effect
    @reactive.event(input.fetch_episode)
    def _():
        for show in shows.get():
            if show['name'] == input.show_name():
                show_id = show['id']
        
        eps = []
        limit = 50
        offset = 0
    
        while True:
            response = sp.show_episodes(show_id=show_id, limit=limit, offset=offset)
            eps.extend(response['items'])
    
            if response['next']:
                offset += limit
            else:
                break
        
        episodes.set(eps)
        names = [ep['name'] for ep in eps]
        
        ui.insert_ui(
            ui.input_select("episode_name", "Select an episode", choices=names),
            selector="#add",
            where="afterEnd",
        )
        
        ui.insert_ui(
            ui.input_select("k", "Expected number of episodes to find", choices=[i for i in range(1,11)]),
            selector="#add",
            where="afterEnd",
        )
        
        ui.insert_ui(
            ui.input_action_button("cluster", "Find the most similar episodes", theme="gradient-blue-indigo"),
            selector="#add",
            where="afterEnd",
        )
    
    @reactive.effect
    @reactive.event(input.cluster)
    def _():
        for ep in episodes.get():
            if ep['name'] == input.episode_name():
                episode.set(ep)
                break
        
        rows = [0]
        cols = []
        data = []

        words = preprocess_text(ep['episode_description'])
        word_count = Counter(words)
        for word, count in word_count.items():
            if word in filtered_words:
                cols.append(filtered_words.index(word))
                data.append(count)
                
        sparse = csr_matrix((data, (rows, cols)), shape=(len(word_count), len(filtered_words)))
        pc.set(svd.transform(sparse)[:,:87])
        label.set(kmeans.predict(pc.get())[0])
    
    @output
    @render_widget
    def pc12():
        if label.get() == None:
            return None
        
        data = pd.DataFrame({'pc1': pc.get()[:,0], 'pc2': pc.get()[:,1]})
        
        fig = px.scatter(data, x='pc1', y='pc2',
                         title="The Selected Episode And The Nearest Episodes")
        fig.update_layout(
            xaxis_title="Principal Components 1",
            yaxis_title="Principal Components 2"
        )
        return fig

#------------------------------------------------------------------------------

app = App(app_ui, server)
