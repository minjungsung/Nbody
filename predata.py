import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class NBodyDataset(Dataset):
    def __init__(self, file, window_size=5):
        with h5py.File(file, "r") as f:
            data = f["positions"][:]
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = NBodyDataset("nbody_data.h5", window_size=5)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)