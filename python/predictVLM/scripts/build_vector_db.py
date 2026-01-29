from predictVLM.src.configs.train_config import TrainConfig
from predictVLM.src.utils.set_seed import set_seed
from predictVLM.src.data.preprocessing import filtering_news, str_to_python, handle_holidays
from predictVLM.src.data.vector_db.build_db import connect_vector_db, create_collection, push_data

import os
import tyro
import pandas as pd
from dotenv import load_dotenv


def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    # news 데이터 전처리
    news_df = pd.read_csv(os.path.join(cfg.data_dir, "news_articles_resources.csv"))
    news_df = filtering_news(news_df)
    news_df['article_embedding'] = news_df['article_embedding'].apply(str_to_python)
    
    date_df = pd.read_csv(os.path.join(cfg.data_dir, "corn_future_price.csv"))
    news_df = handle_holidays(date_df, news_df)
    
    # DB 연결 및 업로드
    load_dotenv()
    
    client = connect_vector_db(os.getenv('VECTOR_DB_HOST'), os.getenv('VECTOR_DB_PORT'))
    create_collection(client, "news")
    push_data(client, "news", news_df)


if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)