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
