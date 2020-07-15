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

    from sklearn.datasets import fetch_news20
    from jange import apps, evaluation
    texts, labels = fetch_news20(return_X_y=True)
    
    steps = apps.classifier(x=texts, y=labels)
    results = evaluation.evaluate_classifier(x=texts, y=labels, operations=steps)
    print(results)

Well that was easy. What just happened? jange abstracts many steps necessary for building common NLP applications in :py:mod:`jange.apps` module. We use :py:meth:`apps.classifier` function to train a model to categorize the news articles into different categories. Under the hood, the input texts are cleaned by removing emails, numbers, hyperlinks etc and then lemmatized. After this basic pre-processing step, a TF-IDF model is trained to extract feature vectors for each input text which is then passed to a :py:class:`sklearn.linear.SGDClassifier` to predict the labels. Take a look at the evaluation results. The accuracy of the model is pretty good. If you don't know what all the stuff in the results mean then don't worry, we'll cover that in another section.

Clustering
----------
Clustering is a process that finds items that are similar to each other and groups them together. For clustering, you don't need to provide labels like you do in classification problems. jange provides a function `cluster` to quickly get started. You need to pass some texts and how many clusters you want to create and everything is done for you automatically.

::

    from sklearn.datasets import fetch_news20
    from jange import apps, evaluation
    texts, labels = fetch_news20(return_X_y=True)
    
    steps = apps.cluster(x=texts, n_clusters=10)
    results = evaluation.evaluate_cluster(x=texts, operations=steps)
    print(results)
