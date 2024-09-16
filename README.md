# Advanced Flexible Profiler

The Advanced Flexible Profiler is a powerful and customizable profiling tool designed to help you analyze and optimize your Python code. It provides granular control over profiling different aspects of your code, including function-level profiling, block profiling, and line-by-line profiling.

## Features

- Function-level profiling
- Block profiling
- Line-by-line profiling
- Flexible configuration options
- Customizable saving and printing of profiling reports
- JSON and TXT output formats
- Easy integration with existing code

## Installation

To use the Advanced Flexible Profiler, clone this repository and include the `improved_profiler.py` and `profiler_config.py` files in your project directory.

```bash
git clone https://github.com/yourusername/advanced-flexible-profiler.git
cd advanced-flexible-profiler
cp improved_profiler.py profiler_config.py /path/to/your/project/
```

## Usage

### Basic Setup

1. Import the profiler in your Python script:

```python
from improved_profiler import profiler, profile_methods, profile_line_by_line, auto_profile_blocks
from profiler_config import profiler_config
```

2. Configure the profiler settings:

```python
profiler_config.enabled = True
profiler_config.output_dir = "profiling_output"
profiler_config.save_format = "json"
profiler_config.steps_to_save = [1, 100, 1000]
profiler_config.print_block_profile = True
profiler_config.save_line_profile = True
profiler_config.save_block_profile = True
profiler_config.save_overall_profile = True
```

### Profiling Examples

#### Example 1: Profiling a Class

```python
@profile_methods
class MyClass:
    def __init__(self):
        self.data = [i for i in range(1000000)]
    
    def process_data(self):
        return sum(self.data)

# Usage
obj = MyClass()
result = obj.process_data()
profiler.print_overall_profile()
```

Output:
```
Overall Profiling Report:
Function Profiling:
MyClass.__init__: Total time: 0.052341s, Calls: 1, Avg time: 0.052341s
MyClass.process_data: Total time: 0.021567s, Calls: 1, Avg time: 0.021567s
```

#### Example 2: Profiling a Specific Block of Code

```python
@auto_profile_blocks
def my_function():
    with profiler.profile_context("data_processing"):
        data = [i for i in range(1000000)]
        result = sum(data)
    return result

# Usage
my_function()
profiler.print_block_profile()
```

Output:
```
Block Profiling Report:
my_function: Total time: 0.075231s, Calls: 1, Avg time: 0.075231s
data_processing: Total time: 0.074123s, Calls: 1, Avg time: 0.074123s
```

#### Example 3: Line-by-Line Profiling

```python
@profile_line_by_line
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

# Usage
fibonacci(20)
profiler.save_line_profile(step="fibonacci")
```

This will save a detailed line-by-line profiling report in the `profiling_output` directory.

#### Example 4: Profiling in a Training Loop

```python
@profile_methods
class Trainer:
    @auto_profile_blocks
    def train(self):
        for epoch in range(10):
            with profiler.profile_context(f"epoch_{epoch}"):
                for batch in range(100):
                    self.train_step(batch)
                    
                    if batch % 10 == 0:
                        profiler.step_profiler()
    
    @auto_profile_blocks
    def train_step(self, batch):
        # Simulating a training step
        time.sleep(0.01)

# Usage
trainer = Trainer()
trainer.train()
profiler.save_overall_profile(step="final")
```

This will save profiling reports at specified steps and a final overall report.

### Customizing Profiler Behavior

You can customize the profiler's behavior by modifying the `profiler_config.py` file or updating the configuration in your code:

```python
profiler_config.print_line_profile = False
profiler_config.print_block_profile = True
profiler_config.print_overall_profile = True
profiler_config.save_format = "txt"
profiler_config.steps_to_save = [1, 50, 100]
```

## Advanced Usage

### Saving Custom Profiles

You can save custom profiles at any point in your code:

```python
profiler.save_block_profile(step="custom_step", filename="custom_block_profile.json")
profiler.save_line_profile(step="custom_step", filename="custom_line_profile.txt")
profiler.save_overall_profile(step="custom_step", filename="custom_overall_profile.json")
```

### Resetting the Profiler

If you want to reset the profiler's data:

```python
profiler.reset()
```

## Best Practices

1. Use `@profile_methods` for classes where you want to profile all methods.
2. Use `@auto_profile_blocks` for individual functions you want to profile.
3. Use `with profiler.profile_context("name"):` for profiling specific blocks of code.
4. Call `profiler.step_profiler()` regularly in loops to save and print reports at specified steps.
5. Use `profiler.save_*_profile()` and `profiler.print_*_profile()` methods for custom profiling actions.

## Contributing

Contributions to the Advanced Flexible Profiler are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
