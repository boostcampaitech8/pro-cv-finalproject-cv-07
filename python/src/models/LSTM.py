import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, output_dim=1, lstm_layers=2, fc_hidden_dim=25, dropout=0.1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out