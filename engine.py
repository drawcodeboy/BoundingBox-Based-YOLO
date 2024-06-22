import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

def train_one_epoch(model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer,
                    data_loader: DataLoader, device: torch.device, epoch: int, epochs: int):
    
    print(f"Epoch: [{epoch:03d}/{epochs:03d}]")
    mean_loss = []
    
    model.train()
    
    for batch_idx, (batches, targets) in enumerate(data_loader, start=1):
        batches = batches.to(device)
        targets = targets.to(device)
        
        # Feed-Forward
        optimizer.zero_grad()
        outputs = model(batches)
        loss = loss_fn(outputs, targets)
        
        # Back Propagation
        loss.backward()
        optimizer.step()
        
        mean_loss.append(loss.item())
        print(f"\rTraining: {100*batch_idx/len(data_loader):.2f}% Loss: {loss.item():.4f}", end="")

    return sum(mean_loss)/len(mean_loss) 

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    pass