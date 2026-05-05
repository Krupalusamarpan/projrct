import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class RULDataset(Dataset):
    def __init__(self, X, y, seq_len=20):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.X) - self.seq_len
    
    def __getitem__(self, idx):
        return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

def train_lstm(X_train, y_train, X_val, y_val, epochs=50, lr=0.001):
    seq_len = 20
    dataset = RULDataset(X_train, y_train, seq_len)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = LSTMModel(X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for seq, target in loader:
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target.unsqueeze(1))
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_seq = torch.FloatTensor(X_val[:seq_len*10].reshape(-1, seq_len, X_val.shape[1]))
        val_pred = model(val_seq)
    
    return model

