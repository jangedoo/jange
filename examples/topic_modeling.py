# %% Load data
from sklearn.datasets import fetch_20newsgroups
from jange import ops, stream, apps

# %%
data = fetch_20newsgroups(shuffle=False, remove=("headers", "footers", "quotes"))

# %%
n = 1000
texts = data["data"][:n]

# %% Print first few documents and their topics

vectorize_op = ops.text.tfidf(max_features=1000, stop_words="english")
cleaning_ops = [
    ops.text.token_filter([[{"POS": "NOUN"}]], keep_matching_tokens=True),
]
ds = stream.DataStream(texts, context=texts)
topics_ds = apps.topic_model(
    ds,
    vectorize_op=vectorize_op,
    cleaning_ops=cleaning_ops,
    max_words_per_topic=6,
    num_topics=100,
)

# %%
for topic, text in zip(topics_ds, topics_ds.context):
    print(text[:400])
    print(topic)
    print("\n\n")


# %%
