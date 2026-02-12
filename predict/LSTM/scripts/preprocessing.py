from predict.LSTM.src.configs.train_config import TrainConfig
from predict.LSTM.src.utils.set_seed import set_seed
from predict.LSTM.src.data.feature_engineering import *
from predict.LSTM.src.data.preprocessing import filtering_news, add_news_sentiment, add_news_timeframe

import os
import csv
import tyro


def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    news_df = pd.read_csv(os.path.join(cfg.data_dir, "news_articles_resources.csv"))

    os.makedirs(os.path.join(cfg.data_dir, "preprocessing"), exist_ok=True)
    
    if cfg.use_news_imformation_features:
        if os.path.exists(os.path.join(cfg.data_dir, "preprocessing/news_imformation.csv")):
            filtering_news_df = pd.read_csv(os.path.join(cfg.data_dir, "preprocessing/news_imformation.csv"))
        else:
            filtering_news_df = filtering_news(news_df)
            filtering_news_df = add_news_sentiment(filtering_news_df)
            filtering_news_df = add_news_timeframe(filtering_news_df)

            filtering_news_df.to_csv(
                os.path.join(cfg.data_dir, "preprocessing/news_imformation.csv"),
                index=False,
                sep=',',
                encoding='utf-8',
                lineterminator='\n',
                quoting=csv.QUOTE_ALL
            )

    for name in ["corn", "wheat", "soybean"]:
        data = pd.read_csv(os.path.join(cfg.data_dir, f"{name}_future_price.csv"))
        data = add_log_return_feature(data, cfg.horizons)
        data = add_ema_features(data, cfg.ema_spans) if cfg.use_ema_features else data
        data = add_volatility_features(data, cfg.volatility_windows) if cfg.use_volatility_features else data
        data = add_news_count_feature(data, news_df) if cfg.use_news_count_features else data
        data = add_news_imformation_features(data, filtering_news_df) if cfg.use_news_imformation_features else data
        data.to_csv(os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv"), index=False)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)