# Machine Learning Training Example with Advanced Flexible Profiler

This example demonstrates how to use the Advanced Flexible Profiler in a machine learning training scenario. The script `ml_training_example.py` implements a simple binary classification task using a neural network and showcases various profiling techniques.

## Overview

The script performs the following steps:
1. Generates dummy data for training and validation
2. Defines a simple neural network model
3. Implements a `Trainer` class with methods for training and validation
4. Runs the training loop for a specified number of epochs

Throughout these steps, the Advanced Flexible Profiler is used to measure performance and provide insights into the execution time of different parts of the code.

## Profiler Configuration

The profiler is configured at the beginning of the script:

```python
profiler_config.enabled = True
profiler_config.output_dir = "ml_profiling_output"
profiler_config.save_format = "json"
profiler_config.steps_to_save = [1, 10, 50, 100]
profiler_config.print_block_profile = True
profiler_config.save_line_profile = True
profiler_config.save_block_profile = True
profiler_config.save_overall_profile = True
```

This configuration enables the profiler, sets the output directory, specifies JSON as the save format, defines steps at which to save profiles, and enables various printing and saving options.

## Profiling Techniques Demonstrated

### 1. Function-level Profiling

The `@profile_methods` decorator is applied to the `Trainer` class:

```python
@profile_methods
class Trainer:
    # ...
```

This profiles all methods in the `Trainer` class, providing timing information for each method call.

### 2. Block Profiling

The `@auto_profile_blocks` decorator is used on specific methods:

```python
@auto_profile_blocks
def generate_data(num_samples=1000, input_dim=10):
    # ...

@auto_profile_blocks
def train_epoch(self, epoch):
    # ...

@auto_profile_blocks
def validate(self):
    # ...
```

This allows for more granular profiling of these specific functions.

### 3. Context-based Profiling

The `profile_context` is used to profile specific blocks of code:

```python
with profiler.profile_context("train_step"):
    # ... training step code ...

with profiler.profile_context(f"epoch_{epoch}"):
    # ... epoch code ...
```

This provides timing information for these specific named blocks of code.

### 4. Line-by-line Profiling

The `@profile_line_by_line` decorator is applied to the main `train` method:

```python
@profile_line_by_line
def train(self, num_epochs):
    # ...
```

This provides detailed timing information for each line in this method.

## Profiler Output

The profiler generates several types of output:

1. **Block Profiles**: Saved after each epoch in JSON format.
2. **Overall Profile**: Saved at the end of training, providing a summary of all profiled functions and blocks.
3. **Line-by-line Profile**: Saved for the `train` method, showing detailed timing for each line.
4. **Terminal Output**: Block profiles are printed to the terminal during training.

## Understanding the Profiler Output

### Block Profiles

Block profiles show the time spent in different blocks of code, such as `train_step`, `epoch_X`, etc. This helps identify which parts of the training process are taking the most time.

### Overall Profile

The overall profile provides a summary of time spent in each profiled function and block across the entire training process. This is useful for identifying overall bottlenecks in your code.

### Line-by-line Profile

The line-by-line profile of the `train` method shows exactly how much time is spent on each line of code within this method. This can be particularly useful for optimizing the main training loop.

## Using the Profiler Output

1. **Identify Bottlenecks**: Look for functions or blocks that take significantly more time than others.
2. **Optimize Data Loading**: If data loading operations are slow, consider using techniques like prefetching or parallel data loading.
3. **GPU Utilization**: For GPU-based training, ensure that the GPU is being fully utilized and that data transfer between CPU and GPU isn't a bottleneck.
4. **Batch Size Optimization**: Experiment with different batch sizes and observe their impact on training time and memory usage.
5. **Model Architecture**: If certain layers in your model are taking too much time, consider simplifying the architecture or using more efficient layer types.

By analyzing the profiler output, you can make data-driven decisions to optimize your machine learning training pipeline for better performance.
