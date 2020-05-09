from absa.config import DATA_PATHS
from absa.dataset import load_dataset
import pandas as pd

train_ds_path = DATA_PATHS['asba.semeval16.raw.train']

df_train = load_dataset(train_ds_path)

print(df_train.shape)

df_train = pd.DataFrame({
    'text': df_train.groupby('id')['text'].first(),
    'categories': df_train.groupby('id')['category'].apply(list),
})

print(df_train.shape)
