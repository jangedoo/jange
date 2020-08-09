.. _tutorial:

Getting started
===============
This tutorial introduces you to basic concepts of working with **jange**. This guide should be enough for you get started with experimenting and deploying.

We'll cover three common NLP applications

- Classification
- Clustering
- Topic Modeling

Classification
--------------
A classification is a problem where we want to assign one or more labels to an input. Let's load the classic newsgroup data-set and train a model to assign categories.
::

    # %% imports
    import logging

    import pandas as pd
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    from jange import ops, stream

    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")

    # %% load data, shuffle and create a stream
    df = pd.read_csv(
        "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv"
    )
    df = df.sample(frac=1.0)  # shuffles the data
    train_df, test_df = train_test_split(df)
    ds = stream.from_df(train_df, columns="news", context_column="type")
    test_ds = stream.from_df(test_df, columns="news", context_column="type")


    # %% train model
    train_preds_ds = ds.apply(
        ops.classify.spacy_classifier(name="classifier"),
        op_kwargs={
            "classifier": {
                "fit_params": {
                    "y": ds.context,
                    "classes": list(set(ds.context)),
                    "n_iter": 10,
                }
            }
        },
    )


    # %% evaluate model
    with ops.utils.disable_training(train_preds_ds.applied_ops) as new_ops:
        preds_ds = test_ds.apply(*new_ops)

    preds = list(preds_ds)
    print(classification_report(test_ds.context, [p.label for p in preds]))


    # %%


Well that was easy. What just happened? jange abstracts many steps necessary for building common NLP applications in :py:mod:`jange.apps` module. We use :py:meth:`apps.classifier` function to train a model to categorize the news articles into different categories. Under the hood, the input texts are cleaned by removing emails, numbers, hyperlinks etc and then lemmatized. After this basic pre-processing step, a TF-IDF model is trained to extract feature vectors for each input text which is then passed to a :py:class:`sklearn.linear.SGDClassifier` to predict the labels. Take a look at the evaluation results. The accuracy of the model is pretty good. If you don't know what all the stuff in the results mean then don't worry, we'll cover that in another section.

Clustering
----------
Clustering is a process that finds items that are similar to each other and groups them together. For clustering, you don't need to provide labels like you do in classification problems. jange provides a function `cluster` to quickly get started. You need to pass some texts and how many clusters you want to create and everything is done for you automatically.

::

    # %% Load data
    from jange import ops, stream, vis

    ds = stream.from_csv(
        "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv",
        columns="news",
        context_column="type",
    )
    print(ds)

    # %% Extract clusters
    # Extract clusters
    result_collector = {}
    clusters_ds = ds.apply(
        ops.text.clean.pos_filter("NOUN", keep_matching_tokens=True),
        ops.text.encode.tfidf(max_features=5000, name="tfidf"),
        ops.cluster.minibatch_kmeans(n_clusters=5),
        result_collector=result_collector,
    )

    # %% Get features extracted by tfidf
    features_ds = result_collector[clusters_ds.applied_ops.find_by_name("tfidf")]

    # %% Visualization
    reduced_features = features_ds.apply(ops.dim.tsne(n_dim=2))
    vis.cluster.visualize(reduced_features, clusters_ds)

