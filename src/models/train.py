import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import our custom architecture
from src.models.pytorch_model import OilPriceNN
from src.data.dataset import OilDataset
from src.data.data_loader import OilDataLoader

def train_model(epochs=100, batch_size=16, learning_rate=0.001):
    """
    The Master Training Pipeline.
    Strictly device-agnostic to seamlessly support both local laptop CPUs 
    and Google Colab's massive NVIDIA GPUs.
    """
    # 1. Device Agnostic Setup (The Colab GPU detector)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Training Pipeline on Hardware: [{device.type.upper()}]")

    # 2. Get the Raw Data
    print("Loading Yahoo Finance data...")
    loader = OilDataLoader()
    loader.fetch_data()
    df = loader.engineer_features()

    # 3. Initialize Warehouse (Dataset) and Delivery Trucks (DataLoader)
    print("Initializing PyTorch Tensor Conversion...")
    dataset = OilDataset(df, target_col='price')
    # Because we use `shuffle=True`, the Neural Network can't memorize the chronological sequence.
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 4. Initialize the Brain
    # Dynamically detect how many feature columns (moving averages, lags) we engineered
    input_size = dataset.features.shape[1] 
    
    # We physically push the empty Neural Network up to the Colab GPU (or local CPU)
    model = OilPriceNN(input_size=input_size).to(device)

    # 5. Define the Rules of Learning
    criterion = nn.MSELoss() # Mean Squared Error (Standard for continuous price prediction)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 6. The Training Loop
    print("\n--- Deep Learning Commenced ---")
    model.train() # Explicitly set model to training mode (enabling dropouts/gradients)
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        # The DataLoader automatically slices our dataset into perfect chunks!
        for batch_index, (features, targets) in enumerate(train_loader):
            
            # Instantly teleport the data rows directly into the GPU memory
            features = features.to(device)
            targets = targets.to(device)
            
            # Step A: Delete old mathematical gradients from the previous batch
            optimizer.zero_grad()
            
            # Step B: Forward Pass (The model attempts to guess the oil prices)
            predictions = model(features)
            
            # Step C: Calculate the Error (How far off were the guesses?)
            loss = criterion(predictions, targets)
            
            # Step D: Backpropagation (The math that actually makes the model smart)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Average Loss (MSE): {epoch_loss/len(train_loader):.4f}")

    # 7. Save the Brain
    os.makedirs("models", exist_ok=True)
    save_path = "models/champion_nn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✅ Training Complete. Enterprise Model weights securely saved to: {save_path}")

if __name__ == "__main__":
    train_model(epochs=100, batch_size=16)
