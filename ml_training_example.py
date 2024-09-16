import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from improved_profiler import profiler, profile_methods, profile_line_by_line, auto_profile_blocks
from profiler_config import profiler_config

# Configure profiler
profiler_config.enabled = True
profiler_config.output_dir = "ml_profiling_output"
profiler_config.save_format = "json"
profiler_config.steps_to_save = [1, 10, 50, 100]
profiler_config.print_block_profile = True
profiler_config.print_line_profile = True
profiler_config.print_overall_profile = True


# Generate dummy data
@profile_line_by_line
def generate_data(num_samples=1000, input_dim=10):
    X = np.random.randn(num_samples, input_dim)
    y = np.sum(X, axis=1) > 0
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


@profile_methods
class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

    @profile_line_by_line
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            with profiler.profile_context("train_step"):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        return total_loss / len(self.train_loader)

    @profile_line_by_line
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)

    @profile_line_by_line
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            with profiler.profile_context(f"epoch_{epoch}"):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        profiler.print_overall_profile()
        profiler.print_block_profile()
        profiler.print_line_profile()


# Main training script
if __name__ == "__main__":
    # Generate data
    X_train, y_train = generate_data(1000, 10)
    X_val, y_val = generate_data(200, 10)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    # Initialize model, criterion, and optimizer
    model = SimpleNN(10)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    # Create trainer and start training
    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader)
    trainer.train(num_epochs=5)
