# Omni Profiler

[Previous content remains the same...]

## Machine Learning Example

Omni Profiler can be particularly useful for profiling machine learning workflows. Here's an example of how to use Omni Profiler with a PyTorch-based neural network training process.

### Setup

First, import the necessary libraries and configure the profiler:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from omni_profiler import profiler, profile_methods, profile_line_by_line, auto_profile_blocks
from profiler_config import profiler_config

# Configure profiler
profiler_config.enabled = True
profiler_config.print_block_profile = True
profiler_config.print_line_profile = True
profiler_config.print_overall_profile = True
```

### Data Generation

We'll use a simple function to generate dummy data:

```python
@auto_profile_blocks
def generate_data(num_samples=1000, input_dim=10):
    X = np.random.randn(num_samples, input_dim)
    y = np.sum(X, axis=1) > 0
    return torch.FloatTensor(X), torch.FloatTensor(y).unsqueeze(1)
```

### Model Definition

Here's a simple neural network model:

```python
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

### Trainer Class

We'll define a `Trainer` class with profiled methods:

```python
@profile_methods
class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, val_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    @auto_profile_blocks
    def train_epoch(self, epoch):
        # Training logic here...
    
    @auto_profile_blocks
    def validate(self):
        # Validation logic here...
    
    @profile_line_by_line
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            with profiler.profile_context(f"epoch_{epoch}"):
                train_loss = self.train_epoch(epoch)
                val_loss = self.validate()
                print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        profiler.print_overall_profile()
```

### Main Training Script

Here's how to put it all together:

```python
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
```

### Profiling Results

After running the training script, Omni Profiler will output detailed profiling information. Here are some example visualizations of the profiling results:

[Image Placeholder: Overall Profiling Report]

[Image Placeholder: Block Profiling Report]

[Image Placeholder: Line-by-Line Profiling Report]

These visualizations help identify performance bottlenecks in your machine learning workflow, allowing you to optimize your code for better efficiency.

## Interpreting Profiling Results

When analyzing the profiling results:

1. Look for functions or blocks that take the most time.
2. Identify any unexpected patterns in the line-by-line profiling.
3. Pay attention to the number of calls for each function or block.
4. Compare the average time per call to identify slow operations.

By using Omni Profiler in your machine learning projects, you can gain valuable insights into the performance characteristics of your training process, helping you optimize your code for faster execution and better resource utilization.

[Rest of the README remains the same...]
