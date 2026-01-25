from src.data.dataset import build_dataset, test_split
from src.data.preprocessing import scale_test_data
from src.models.LSTM import LSTM
from src.engine.inference import test
from src.metrics.metrics import compute_regression_metrics
from src.utils.visualization import save_log_return_plot, save_close_plot
from src.data.postprocessing import convert_close

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# 파라미터 설정
seq_length = 20
batch_size = 64
hidden_dim = 50
num_layers = 2

for name in ["corn", "wheat", "soybean"]:
    if os.path.exists(f"./src/datasets/preprocessing/{name}_feature_engineering.csv"):
        data = pd.read_csv(f"./src/datasets/preprocessing/{name}_feature_engineering.csv")
        dataX, dataY, dataT = build_dataset(data, seq_length)
        test_dates, testX, testY = test_split(dataX, dataY, dataT, "./src/datasets/rolling_fold.json")
        
        checkpoint = torch.load(f'./src/outputs/{name}_best_model.pt', weights_only=False)
        scaler_x = checkpoint["scaler_x"]
        scaler_y = checkpoint["scaler_y"]
        
        testX, testY = scale_test_data(scaler_x, scaler_y, testX, testY)
    
        test_dataset = TensorDataset(testX, testY)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        input_dim = testX.shape[-1]
        output_dim = testY.shape[-1]
        
        model = LSTM(input_dim, hidden_dim, output_dim, num_layers)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        preds, trues = test(model, test_dataloader)
        
        pred_inverse = scaler_y.inverse_transform(np.array(preds))
        testY_inverse = scaler_y.inverse_transform(np.array(trues))
        
        compute_regression_metrics(testY_inverse, pred_inverse)
        
        save_log_return_plot(testY_inverse, pred_inverse, './src/outputs/', filename=f"{name}_log_return_plot.png")
        
        true_close, pred_close = convert_close(data, test_dates, testY_inverse, pred_inverse)
        save_close_plot(test_dates, true_close, pred_close, './src/outputs/', filename=f"{name}_close_plot.png") 
    else:
        print("preprocessing.py 실행 필요")