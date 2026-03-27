import torch
from torch.utils.data import Dataset
import pandas as pd

class OilDataset(Dataset):
    """
    The 'Warehouse Manager'. 
    Takes a perfectly engineered Pandas DataFrame from OilDataLoader 
    and translates it into PyTorch Tensors on-demand.
    """
    def __init__(self, data: pd.DataFrame, target_col: str = 'price'):
        """
        1. Initialization (The Map)
        We receive the Pandas DataFrame. We instantly split it into Features (X) 
        and the Target we want to predict (y).
        """
        # Store all columns EXCEPT the target as our feature matrix
        self.features = data.drop(columns=[target_col]).values
        
        # Store the target column as our answers
        self.targets = data[target_col].values
        
    def __len__(self):
        """
        2. Capacity
        PyTorch needs to know exactly how many rows are in the warehouse.
        """
        return len(self.targets)
        
    def __getitem__(self, idx):
        """
        3. Retrieval
        When the DataLoader truck asks for box `idx`, we must return 
        that specific row as a tuple of (Feature_Tensor, Target_Tensor).
        """
        # We enforce torch.float32 because that is the default mathematical precision for standard Neural Networks
        x_tensor = torch.tensor(self.features[idx], dtype=torch.float32)
        
        # We wrap the target in a list [ ] to ensure it has a shape of (1,) instead of being a scalar
        y_tensor = torch.tensor([self.targets[idx]], dtype=torch.float32)
        
        return x_tensor, y_tensor
