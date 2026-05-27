import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import MultiOutputLSTM
from utils import preprocess_data

# Configurations
INPUT_DIM = 11
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 55 # 11 features * 5 days
BATCH_SIZE = 32
EPOCHS = 150
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    X, y, scaler = preprocess_data('BAJFINANCE.csv')
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.FloatTensor(y).reshape(y.shape[0], -1).to(DEVICE)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = MultiOutputLSTM(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS, OUTPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Starting Training on {DEVICE}...")
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.6f}')

if __name__ == "__main__":
    train()
