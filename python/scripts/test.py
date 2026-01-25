from src.configs.train_config import TrainConfig
from src.utils.set_seed import set_seed
from src.data.dataset import build_dataset, test_split
from src.data.preprocessing import scale_test_data
from src.models.LSTM import LSTM
from src.engine.inference import test
from src.metrics.metrics import compute_regression_metrics
from src.utils.visualization import save_log_return_plot, save_close_plot
from src.data.postprocessing import convert_close

import os
import tyro
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(cfg: TrainConfig):
    set_seed(cfg.seed)

    for name in ["corn", "wheat", "soybean"]:
        data_path = os.path.join(cfg.data_dir, f"preprocessing/{name}_feature_engineering.csv")
        checkpoint_path = os.path.join(cfg.checkpoint_dir, f"{name}_{cfg.seq_length}window_best.pt")
        
        if not os.path.exists(data_path):
            print(f"{data_path} 파일 존재하지 않음 \n preprocessing.py 실행 필요")
            return
        
        if not os.path.exists(checkpoint_path):
            print(f"{checkpoint_path} 파일 존재하지 않음 \n train.py 실행 필요")
            return 
    
        data = pd.read_csv(data_path)
        dataX, dataY, dataT = build_dataset(data, cfg.seq_length, cfg.horizons)
        test_dates, testX, testY = test_split(dataX, dataY, dataT, os.path.join(cfg.data_dir, "rolling_fold.json"))
        
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        scaler_x = checkpoint["scaler_x"]
        scaler_y = checkpoint["scaler_y"]
        
        testX, testY = scale_test_data(scaler_x, scaler_y, testX, testY)
    
        test_dataset = TensorDataset(testX, testY)

        test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        
        input_dim = testX.shape[-1]
        output_dim = testY.shape[-1]
        
        model = LSTM(input_dim, cfg.hidden_dim, output_dim, cfg.num_layers)
        model.load_state_dict(checkpoint["model_state_dict"])
        
        preds, trues = test(model, test_dataloader)
        
        print(preds.shape)
        print(trues.shape)
        
        if output_dim == 1:
            pred_inverse = scaler_y.inverse_transform(np.array(preds))
            testY_inverse = scaler_y.inverse_transform(np.array(trues))
        else:
            pred_inverse = scaler_y.inverse_transform(np.array(preds).reshape(-1, len(preds[0])))
            testY_inverse = scaler_y.inverse_transform(np.array(trues).reshape(-1, len(trues[0])))
        
        compute_regression_metrics(testY_inverse, pred_inverse, cfg.horizons)
        
        save_log_return_plot(testY_inverse, pred_inverse, cfg.horizons, os.path.join(cfg.output_dir, f"{name}"), "log_return_plot.png")
        
        true_close, pred_close = convert_close(data, test_dates, testY_inverse, pred_inverse, cfg.horizons)
        save_close_plot(test_dates, true_close, pred_close, cfg.horizons, os.path.join(cfg.output_dir, f"{name}"), "close_plot.png") 
        

if __name__ == "__main__":
    cfg = tyro.cli(TrainConfig)
    main(cfg)