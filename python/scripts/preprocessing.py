from src.utils.set_seed import set_seed
from src.data.feature_engineering import *
from src.data.preprocessing import filtering_news, add_news_sentiment, add_news_timeframe

import os
import csv


set_seed(42)

news_df = pd.read_csv("./src/datasets/news_articles_resources.csv")

os.makedirs("./src/datasets/preprocessing", exist_ok=True)
if os.path.exists("./src/datasets/preprocessing/news_imformation.csv"):
    filtering_news_df = pd.read_csv("./src/datasets/preprocessing/news_imformation.csv")
else:
    filtering_news_df = filtering_news(news_df)
    filtering_news_df = add_news_sentiment(filtering_news_df)
    filtering_news_df = add_news_timeframe(filtering_news_df)

    filtering_news_df.to_csv(
        "./src/datasets/preprocessing/news_imformation.csv",
        index=False,
        sep=',',
        encoding='utf-8',
        lineterminator='\n',
        quoting=csv.QUOTE_ALL
    )

for name in ["corn", "wheat", "soybean"]:
    data = pd.read_csv(f"./src/datasets/{name}_future_price.csv")
    data = add_log_return_feature(data)
    data = add_ema_features(data)
    data = add_volatility_features(data)
    data = add_news_count_feature(data, news_df)
    data = add_news_imformation_features(data, filtering_news_df)
    data.to_csv(f"./src/datasets/preprocessing/{name}_feature_engineering.csv", index=False)