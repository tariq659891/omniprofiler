# Omni Profiler

Omni Profiler is a powerful and flexible profiling tool designed to help you analyze and optimize your Python code. It provides granular control over profiling different aspects of your code, including function-level profiling, block profiling, and line-by-line profiling.

## Features

- Function-level profiling
- Block profiling
- Line-by-line profiling
- Flexible configuration options
- Customizable printing of profiling reports
- Tabular output format
- Easy integration with existing code

## Installation

To use Omni Profiler, clone this repository and include the necessary files in your project directory.

```bash
git clone https://github.com/yourusername/omni-profiler.git
```

Make sure to install the required dependencies:

```bash
pip install line_profiler tabulate
```

## Usage

### Basic Setup

1. Import the profiler in your Python script:

```python
from omni_profiler import profiler, profile_methods, profile_line_by_line, auto_profile_blocks
from profiler_config import profiler_config
```

2. Configure the profiler settings:

```python
profiler_config.enabled = True
profiler_config.print_line_profile = True
profiler_config.print_block_profile = True
profiler_config.print_overall_profile = True
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
profiler.print_line_profile()
```

### Customizing Profiler Behavior

You can customize the profiler's behavior by modifying the `profiler_config.py` file or updating the configuration in your code:

```python
profiler_config.enabled = True
profiler_config.print_line_profile = True
profiler_config.print_block_profile = True
profiler_config.print_overall_profile = True
```

## Advanced Usage

### Getting Profiling Data

You can get profiling data programmatically:

```python
line_profile_data = profiler.get_line_profile()
block_profile_data = profiler.get_block_profile()
overall_profile_data = profiler.get_overall_profile()
```

### Printing Profiling Reports

The profiler provides methods to print different types of profiling reports:

```python
profiler.print_line_profile()
profiler.print_block_profile()
profiler.print_overall_profile()
```

## Best Practices

1. Use `@profile_methods` for classes where you want to profile all methods.
2. Use `@profile_line_by_line` for functions where you need detailed line-by-line profiling.
3. Use `@auto_profile_blocks` for individual functions you want to profile.
4. Use `with profiler.profile_context("name"):` for profiling specific blocks of code.
5. Configure the profiler using `profiler_config` to control what data is collected and displayed.

## Future Work

We have exciting plans to enhance Omni Profiler in the future:

1. Implement saving functionality to store profiling results in various formats (JSON, CSV, etc.).
2. Add support for asynchronous code profiling.
3. Develop a graphical user interface for easier visualization of profiling results.
4. Integrate with popular IDEs for seamless profiling experience.
5. Implement memory profiling capabilities.
6. Add support for distributed system profiling.

Stay tuned for these upcoming features!

## Contributing

Contributions to Omni Profiler are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

Special thanks to the creators of the `line_profiler` package, which forms the backbone of our line-by-line profiling functionality. Their work has been instrumental in making this profiler possible.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
