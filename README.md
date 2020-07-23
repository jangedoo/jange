# jange
[![Build Status](https://travis-ci.org/jangedoo/jange.svg?branch=master)](https://travis-ci.org/jangedoo/jange)
------
jange is an easy to use NLP library for Python. It provides a high level API for commonly applications in NLP. It is based on popular libraries like `pandas`, `scikit-learn`, `spacy`, and `plotly`.

To get started, install using pip.
```
pip install jange
```

# Overview
For common NLP applications, we clean the data, extract the features and apply some ML algorithm on the features.We apply some transformation on the raw input data to get the results we want. **jange** organizes these transformations as a series of operation we do on the input. The high level API for these transformation are easy to read and reason with. Take a look at the example below for clustering. Even without any explanation you should be able to read and understand what is happening without refering to a hour length tutorial or trying to wrap your head around multi-dimensional array slicing and dicing. Let's not forget the pain to migrate the code from prototyping to production. **jange** tries to simplify the transition from experimental phase to production use.


```python
# %% Load data
from jange import ops, stream, vis

ds = stream.from_csv(
    "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv",
    columns="news",
    context_column="type",
)

# %% Extract clusters
# Extract clusters
result_collector = {}
clusters_ds = ds.apply(
    ops.text.clean.filter_pos("NOUN", keep_matching_tokens=True),
    ops.text.encode.tfidf(max_features=5000, name="tfidf"),
    ops.cluster.minibatch_kmeans(n_clusters=5),
    result_collector=result_collector,
)

# %% Get features extracted by tfidf
features_ds = result_collector[clusters_ds.applied_ops.find_by_name("tfidf")]

# %% Visualization
reduced_features = features_ds.apply(ops.dim.tsne(n_dim=2))
vis.cluster.visualize(reduced_features, clusters_ds)

# visualization looks good, lets export the operations
with ops.utils.disable_training(cluster_ds.applied_ops) as cluster_ops:
    with open("cluster_ops.pkl", "wb") as f:
        pickle.dump(cluster_ops, f)

# in_another_file.py
# load the saved operations and apply on a new stream to retrieve the clusters
with open("cluster_ops.pkl", "rb") as f:
    cluster_ops = pickle.load(f)

clusters_ds = input_ds.apply(cluster_ops)
```
![Cluster](https://sanjayasubedi.com.np/assets/images/nlp/clustering/cluster_jange.png)



Looks convincing?

# What can jange do for me?
The idea behind jange is for rapid prototyping and **deployment**. Jange supports

- Data cleaning: remove stop words, emails, links, numbers, filter tokens based on POS or any filter operation using spacy's token matcher patterns. It provides a high-level api to spacy's TokenMatcher.
- Text Encoding : Provides high level API for encoding texts as one-hot, count or tf-idf features using scikit-learn model
- Embedding : Document embedding based on spacy's language model that captures semantics of the text
- Clustering: High level API for several clustering algorithsm in scikit-learn library
- Topic modeling: High level API for commonly used topic modeling algorithms (NMF, LDA)
- Nearest Neighbors : High level API for finding similar pair or groups of similar items
- Classification : High level API to train spacy's model or many of scikit-learn's classifiers
- Dimension reduction: High level API for algorithms used to reduce dimension of feature space. Useful for visualization (tsne, pca) or compression
- Extraction : High level API to extract sentences, or summary from texts
And many more including visualization, operation persistence and quick apps.

# Basic Concept
## DataStream
DataStream is a holder of your data. The data can be lazily loaded or can be in memory. A DataStream is nothing more than a list of items together with some context(optional). For example,
```python
from jange import stream
# create stream from any python object including lists, numpy array, generators etc.
ds = stream.DataStream(items=["Product 1", "Product 2"], context=["pid1", "pid2"])
# few helper functions
stream.from_csv("path/to/csv")
stream.from_df(df)
```
`ds` is a data stream that holds your data along with some context. In this case the database id of the products. The idea behind context is that it holds some metadata about the items you pass. If you don't pass anything to the context, then jange will internally create context values for each item. DataStream also maintains information about what operations have been applied to it in a variable `applied_ops`. For example, if you applied few cleaning, one tf-df and a topic modeling operation to an input stream, the final `DataStream` containing the output of topic modeling will know about all operations that were applied from the beginning. You can apply the same operations to a new raw input stream and exactly the same operations will be applied to the new input.


## Operations
Transformations to the input data are done by `Operation` in jange. Each operation takes in a DataStream and produces a DataStream. One or more operations are applied to a DataStream. Each operation will execute and pass the results to the next operation. Example below shows how you can apply 
different operations. Of course, the output of an operation should be compatible with the input the next operation is expecting.

Operations in **jange** are available under `ops` sub package. They are nicely organized into modules depending on their scope. For example, operations that work on input of texts are under `ops.text`. For cleaning the operations are under `ops.text.clean` and for encoding texts into vectors or embeddings, `ops.text.encode` or `ops.text.embedding` can be used.

For clustering, topic modeling, classification etc. they can be found under `ops.cluster`, `ops.topic`, `ops.classfy` etc.
```python
input_ds = stream.from_csv(
    "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv",
    columns="news",
    context_column="type",
)
clusters_ds = input_ds.apply(
    ops.text.clean.filter_pos("NOUN", keep_matching_tokens=True),
    ops.text.encode.tfidf(max_features=5000, name="tfidf"),
    ops.cluster.minibatch_kmeans(n_clusters=5)
)

# once we are happy with results save the operations to disk
with ops.utils.disable_training(cluster_ds.applied_ops) as cluster_ops:
    with open("cluster_ops.pkl", "wb") as f:
        pickle.dump(cluster_ops, f)

# in_another_file.py
# load the saved operations and apply on a new stream to retrieve the clusters
with open("cluster_ops.pkl", "rb") as f:
    cluster_ops = pickle.load(f)

clusters_ds = input_ds.apply(cluster_ops) # WOW! this easy for production? 👍
```

### How does it work?
`Operation` has a very simple interface with one method `run(ds: DataStream) -> DataStream`. When it processes input DataStream, and produces an output, it will add itself and the `applied_ops` of input DataStream to the `applied_ops` of output DataStream. From the code above, if we print out `cluster_ds.applied_ops`, we'll see a list of 3 operations so we know exactly what operations were applied to produce this output. Each operation will also make sure the context is passed to the output appropriately. This is important when some operations discard one or more items from the input. If you solely rely on array indexing, the mapping of output to the original input index will no longer be valid as some items have been removed and you don't know which output maps to which original input anymore. Context helps to maintain the mapping with original data.

 What about operations where we need to train? These operations use `TrainableMixin` which has an attribute `should_train`. By default it is True, so when you run it, any trainable operation will train the underlying model (sklearn's or spacy's models) and then do the predictions. But during production, you don't want to train so a helper function `ops.utils.disable_training` will set `should_train` to False for all trainable operations. As shown in the example above, you can save these operations to disk and next time you load it, you can run these operations with out training any "trainable" operations again.

Care has been taken in making sure that the operations can be pickled without saving unnecessary data. For example, instead of pickling spacy's language model, the operation only saves the path to the model and when operation is unpicked, spacy's model is loaded from that path. Also, the model loading is cached, so attempts to load the same spacy model will use the cached version instead of loading from the disk.

Since the API is so simple, you can easily extend to fit your requirements.

Check out the [API Reference](https://jange.readthedocs.io/en/latest/api/index.html) for more details.

# Installation
```
pip install jange
```

## From source
This project uses poetry to manage dependencies. If you don't already have poetry installed then go to https://python-poetry.org/docs/#installation for instructions on how to install it for your OS.

Once poetry is installed, from the root directory of this project, run `poetry install`. It will create a virtual environment for this project and install the necessary dependencies (including dev dependencies).


# Contributions 👩‍💻
This library is in a very early stage. Your perspective on how things could be done or improved would be greatly appreciated. Since this is in early stage, you'll most probably encounter some bugs and issues. Please let us know by opening an issue or if you know Python then you can contribute too!