# Spotify_episodes_clustering
We provide three folders in our repository. 

- Code folder includes 4 ipynb files.

-- data.ipynb provides step to extract episodes and shows descriptions data using Spotify API. You need to provide your Spotify client_id and client_secret to extract data. Also you can chose epoch. Each epoch can extract about 5,000 shows data and 100,000 episodes data.

pca_train.ipynb provides step to train PCA factor loadings.

pca_computing.ipynb provides steps to compute the PC value for each episodes data.

clustering.ipynb provides steps to cluster the data of PC values from pca_computing.ipynb.
