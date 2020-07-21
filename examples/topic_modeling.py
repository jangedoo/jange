# %% Load imports
from jange import apps, ops, stream

# %% Load sample data

ds = stream.from_csv(
    "https://raw.githubusercontent.com/jangedoo/jange/master/dataset/bbc.csv",
    columns="news",
    context_column="news",
)
print(ds)

# %% Run topic modeling

cleaning_ops = [
    ops.text.clean.filter_pos("NOUN", keep_matching_tokens=True),
    ops.text.clean.lemmatize(),
]

vectorize_op = ops.text.encode.tfidf(max_features=5000, stop_words="english")
topics_ds = apps.topic_model(
    ds,
    vectorize_op=vectorize_op,
    cleaning_ops=cleaning_ops,
    max_words_per_topic=6,
    num_topics=100,
)

# %%
for i, (topic, text) in enumerate(zip(topics_ds, topics_ds.context)):
    print(text[:100])
    print(topic)
    print("\n\n")
    if i > 100:
        break


# %%
