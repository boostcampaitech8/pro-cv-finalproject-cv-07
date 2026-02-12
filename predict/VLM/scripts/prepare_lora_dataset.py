from predict.VLM.src.configs.train_config import TrainConfig
from predict.VLM.src.utils.set_seed import set_seed
from predict.VLM.src.data.preprocessing import (
    filtering_price,
    add_price,
    add_initial_prompt,
    add_embedding,
    add_relative_news,
    add_final_prompt,
    add_output,
    train_valid_test_split,
    add_prev_close
)

import os
import json
import tyro
import pandas as pd
from dotenv import load_dotenv


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    load_dotenv()

    if os.path.exists(os.path.join(cfg.data_dir, f"preprocessing/{cfg.name}_window{cfg.seq_length}.json")):
        with open(os.path.join(cfg.data_dir, f"preprocessing/{cfg.name}_window{cfg.seq_length}.json"), 'r') as file:
            data = json.load(file)
    else:
        price_df = pd.read_csv(os.path.join(cfg.data_dir, f"{cfg.name}_future_price.csv"))
    
        # date dict 생성
        date_list = sorted(price_df['time'].unique())
        valid_date_list = sorted(filtering_price(price_df)['time'].unique())
        data = [{"date": d} for d in valid_date_list]
        
        # horizons 설정
        horizons = [i + 1 for i in range(cfg.horizons)]
    
        # data 추가
        add_price(data, price_df, cfg.seq_length, cfg.ema_spans)
        add_initial_prompt(data, cfg.name, cfg.seq_length, horizons, cfg.ema_spans)
        add_embedding(data, os.getenv('AWS_ACCESS_KEY_ID'), os.getenv('AWS_SECRET_ACCESS_KEY'))
        add_relative_news(data, date_list, os.getenv('VECTOR_DB_HOST'), os.getenv('VECTOR_DB_PORT'))
        add_final_prompt(data, cfg.name, cfg.seq_length, horizons, cfg.ema_spans)
        data = add_output(data, date_list, price_df, horizons)
        add_prev_close(data, date_list, price_df, horizons)
    
        # 저장
        os.makedirs(os.path.join(cfg.data_dir, "preprocessing"), exist_ok=True)
    
        with open(os.path.join(cfg.data_dir, f"preprocessing/{cfg.name}_window{cfg.seq_length}.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(cfg.data_dir, 'rolling_fold.json'), 'r') as file:
        split_file = json.load(file)
    
    train_valid_test_split(data, split_file, cfg.fold, cfg.data_dir, cfg.image_dir)
    

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)