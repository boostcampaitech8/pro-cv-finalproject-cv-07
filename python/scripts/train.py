from src.data.dataset import *
from src.data.preprocessing import scale_train_data
from src.models.LSTM import LSTM
from src.engine.trainer import train
from src.utils.visualization import save_loss_curve

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# 파라미터 설정
seq_length = 20
batch_size = 64
hidden_dim = 50
learning_rate = 0.001
num_layers = 2
num_epochs = 300

for name in ["corn", "wheat", "soybean"]:
    if os.path.exists(f"./src/datasets/preprocessing/{name}_feature_engineering.csv"):
        data = pd.read_csv(f"./src/datasets/preprocessing/{name}_feature_engineering.csv")
        dataX, dataY, dataT = build_dataset(data, seq_length)
        trainX, trainY, validX, validY = train_valid_split(dataX, dataY, dataT, "./src/datasets/rolling_fold.json", index=7)
        scaler_x, scaler_y, trainX, trainY, validX, validY = scale_train_data(trainX, trainY, validX, validY)
    
        train_dataset = TensorDataset(trainX, trainY)
        valid_dataset = TensorDataset(validX, validY)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
        input_dim = trainX.shape[-1]
        output_dim = trainY.shape[-1]
        
        model = LSTM(input_dim, hidden_dim, output_dim, num_layers)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        model, train_hist, valid_hist = train(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs)
        
        checkpoint = {"model_state_dict": model.state_dict(), "scaler_x": scaler_x, "scaler_y": scaler_y,}
        torch.save(checkpoint, f'./src/outputs/{name}_best_model.pt')
        
        save_loss_curve(train_hist, valid_hist, './src/outputs/', filename=f"{name}_loss_curve.png")
        
    else:
        print("preprocessing.py 실행 필요")