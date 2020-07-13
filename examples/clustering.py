# %% Load data
from sklearn.datasets import fetch_20newsgroups
import spacy
from jange import ops, stream, vis

data = fetch_20newsgroups(shuffle=False)
n = 100
texts = data["data"][:n]
label_names = dict((enumerate(data["target_names"])))
labels = [label_names[i] for i in data["target"][:n]]

# %% Load spacy model
nlp = spacy.load("en_core_web_md")

# %% Extract features
input_ds = stream.DataStream(items=texts, context=labels)
features_ds = input_ds.apply(
    ops.text.remove_emails(nlp),
    ops.text.remove_links(nlp),
    ops.text.tfidf(max_features=5000),
)

# %% Extract clusters
clusters_ds = features_ds.apply(ops.cluster.minibatch_kmeans(n_clusters=4))
print(len(clusters_ds.items))

# %% Visualization
reduced_features = features_ds.apply(ops.dim.pca(n_dim=3))
vis.cluster.visualize(reduced_features, clusters_ds)


# %%

