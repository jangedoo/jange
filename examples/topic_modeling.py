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
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# %% Extract features
input_ds = stream.DataStream(items=texts, context=labels)
features_ds = input_ds.apply(
    ops.text.remove_numbers(nlp),
    ops.text.remove_emails(nlp),
    ops.text.remove_links(nlp),
    ops.text.token_filter([[{"POS": "NOUN"}]], nlp=nlp, keep_matching_tokens=True),
    ops.text.lemmatize(nlp=nlp),
    ops.text.tfidf(max_df=0.95, min_df=0.02),
)

# %% Extract topics
feature_names = features_ds.applied_ops.find_by_name("tfidf").model.get_feature_names()
topics_ds = features_ds.apply(ops.topic.nmf(n_topics=5, feature_names=feature_names))

# %% Print first few documents and their topics
for topic, text in zip(topics_ds, texts[:30]):
    print(text[200:400])
    print(topic)
    print("\n\n")
