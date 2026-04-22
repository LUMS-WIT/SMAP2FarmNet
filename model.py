# model.py
import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=16, num_layers=1, output_residual=True):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.output_residual = output_residual

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def make_model(**cfg):
    return LSTM(
        input_dim=cfg.get('input_dim', 1),
        hidden_dim=cfg.get('hidden_dim', 16),
        num_layers=cfg.get('num_layers', 1),
        output_residual=cfg.get('output_residual', True)
    )


# ===============================================================================
# Model with MC Dropout for Uncertainty Estimation
# ===============================================================================
class LSTM_MC(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, num_layers=1, dropout=0.2, output_residual=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.output_residual = output_residual

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out


def make_model_mc(**cfg):
    return LSTM_MC(
        input_dim=cfg.get('input_dim', 1),
        hidden_dim=cfg.get('hidden_dim', 32),
        num_layers=cfg.get('num_layers', 1),
        dropout=cfg.get('dropout', 0.2),
        output_residual=cfg.get('output_residual', True)
    )