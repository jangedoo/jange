Welcome to jange's documentation!
=================================

Welcome to **jange**, a hasslefree NLP library built with two things in mind:

- easy experiments with less boilerplate
- easy deployment for production use case


**jange** is powered by "industrial strength" NLP capabilities of `spacy` and battle-tested `scikit-learn` for different machine learning algorithms for clustering, classification, topic modeling and more.

Why?
====
Most of the NLP projects I've done have very similar kind of setup:

* load data
* clean
* vectorize
* train clustering or classification model
* evaluate results
* if ok then export the models/associated files otherwise repeat from `clean`
* load exported binaries in a "production package" and serve the requests

As you know while experimenting, most of the time will be spent cleaning, vectorizing and training. But once you are satisfied with the results it can take quite a lot of effort to deploy it to production.

**jange** aims to represent the steps involved from cleaning to evaluating and using it in the production to be as simple as possible.

Not convinced? Take a look at an example below

Example
=======
```python
from sklearn.datasets import fetch_20newsgroups
from jange import stream, ops, vis

texts, labels = fetch_20newsgroups(shuffle=False, return_X_y=True)

# create a data stream
ds = DataStream(items=texts)

# apply operations to the stream to get clusters
clusters_ds = ds.apply(
   ops.text.remove_emails(),
   ops.text.remove_links(),
   ops.text.tfidf(max_features=5000),
   ops.cluster.minibatch_kmeans(n_clusters=4)
)

for text_idx, cluster in zip(clusters_ds.context, clusters_ds):
   print(texts[text_idx][200:500]) # discard email header and limit to 300 chars
   print(f"Cluster = {cluster}")
   print("\n\n")
```

Installation
============
Using pip is recommended. All necessary dependencies will be installed.

``pip install jange``

`spacy` will be installed however you will need to download the language models seperately. For example to download a small English spacy model you need to run

``python -m spacy download en_core_web_sm``

Visit https://spacy.io/models for available models in various sizes and languages.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`