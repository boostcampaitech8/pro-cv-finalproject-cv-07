from predict.LSTM.src.configs.train_config import TrainConfig
from predict.LSTM.src.utils.set_seed import set_seed
from predict.LSTM.src.data.dataset import build_dataset, train_valid_split
from predict.LSTM.src.data.preprocessing import scale_train_data
from predict.LSTM.src.models.LSTM import LSTM
from predict.LSTM.src.engine.trainer import train
from predict.LSTM.src.utils.visualization import save_loss_curve

import os
import tyro
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def main(cfg: TrainConfig):
    set_seed(cfg.seed)
    
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    for name in ["corn", "wheat", "soybean"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        
        if not os.path.exists(data_path):
            print(f"{data_path} 파일 존재하지 않음 \n preprocessing.py 실행 필요")
            return
        
        data = pd.read_csv(data_path)
        dataX, dataY, dataT = build_dataset(data, cfg.seq_length, cfg.horizons)
        
        for fold in cfg.fold:
            trainX, trainY, validX, validY = train_valid_split(dataX, dataY, dataT, os.path.join(cfg.data_dir, "rolling_fold.json"), index=fold)
            scaler_x, scaler_y, trainX, trainY, validX, validY = scale_train_data(trainX, trainY, validX, validY)
    
            train_dataset = TensorDataset(trainX, trainY)
            valid_dataset = TensorDataset(validX, validY)
        
            train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
    
            input_dim = trainX.shape[-1]
            output_dim = trainY.shape[-1]
        
            model = LSTM(input_dim, cfg.hidden_dim, output_dim, cfg.num_layers)

            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            criterion = nn.MSELoss()
        
            model, train_hist, valid_hist = train(model, train_dataloader, valid_dataloader, criterion, optimizer, cfg.epochs)
        
            checkpoint = {"model_state_dict": model.state_dict(), "scaler_x": scaler_x, "scaler_y": scaler_y}
            torch.save(checkpoint, os.path.join(cfg.checkpoint_dir, f"{name}_{cfg.seq_length}window_{fold}flod_best.pt"))

            os.makedirs(os.path.join(cfg.output_dir, f"{name}"), exist_ok=True)
            save_loss_curve(train_hist, valid_hist, os.path.join(cfg.output_dir, f"{name}"), f"{cfg.seq_length}window_{fold}flod_loss_curve.png")
            

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)