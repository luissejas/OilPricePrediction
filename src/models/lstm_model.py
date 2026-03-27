import torch
import torch.nn as nn

class OilPriceLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) Architecture for Sequential Market Prediction.
    Significantly more powerful than a standard NN because it biologically remembers 
    the chronology of events rather than viewing each day in isolation.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(OilPriceLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The actual Recurrent Memory Engine
        # batch_first=True strictly enforces the format: (Batch_Size, Sequence_Length, Features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # The Translation Layer: Converts the vast hidden states into exactly 1 price prediction
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # 1. Dimension Verification
        # Our DataLoader outputs (batch, features) -> e.g., (32, 5)
        # Deep LSTMs demand a sequence dimension: (32, 1, 5)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        # 2. Sequential Logic Processing
        # (hn, cn) are the actual memory cells, we don't strictly need to manually manage them for this architecture
        lstm_out, (hn, cn) = self.lstm(x)
        
        # 3. Harvest the Final Prediction
        # We only care about the very last chronological time-step the LSTM processed
        last_time_step = lstm_out[:, -1, :]
        
        # 4. Final Output Conversion
        out = self.fc(last_time_step)
        return out
