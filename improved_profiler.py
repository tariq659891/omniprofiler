import time
from functools import wraps
from collections import defaultdict
import os
from contextlib import contextmanager
from profiler_config import profiler_config
from tabulate import tabulate
from line_profiler import LineProfiler
import json
import linecache

print("Hello")

class LineProfilerBasedProfiler:
    def __init__(self):
        self.line_profiler = LineProfiler()
        self.function_times = defaultdict(float)
        self.function_calls = defaultdict(int)
        self.block_times = defaultdict(float)
        self.block_calls = defaultdict(int)
        self.current_blocks = []
        self.block_start_times = {}

    def profile_methods(self, cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value) and not attr_name.startswith("__"):
                setattr(cls, attr_name, self.profile_function(attr_value))
        return cls

    def profile_function(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler_config.enabled:
                return func(*args, **kwargs)
            start_time = time.time()
            result = self.line_profiler(func)(*args, **kwargs)
            end_time = time.time()
            self.function_times[func.__name__] += end_time - start_time
            self.function_calls[func.__name__] += 1
            return result
        return wrapper

    def profile_line_by_line(self, func):
        return self.profile_function(func)

    @contextmanager
    def profile_context(self, name):
        if not profiler_config.enabled:
            yield
            return
        start_time = time.time()
        full_name = f"{':'.join(self.current_blocks)}:{name}" if self.current_blocks else name
        self.current_blocks.append(name)
        try:
            yield
        finally:
            self.current_blocks.pop()
            end_time = time.time()
            self.block_times[full_name] += end_time - start_time
            self.block_calls[full_name] += 1

    def get_line_profile(self):
        stats = self.line_profiler.get_stats()
        output = []
        for (filename, line_number, function_name), (nhits, time) in stats.timings.items():
            output.append({
                "Function": function_name,
                "Line #": line_number,
                "Hits": nhits,
                "Time (µs)": time,
                "Per Hit (µs)": time / nhits if nhits > 0 else 0,
            })
        return output

    def get_block_profile(self):
        return [
            {
                "Block Name": block_name,
                "Total Time (s)": total_time,
                "Calls": self.block_calls[block_name],
                "Avg Time (s)": total_time / self.block_calls[block_name] if self.block_calls[block_name] > 0 else 0
            } for block_name, total_time in self.block_times.items()
        ]

    def get_overall_profile(self):
        function_profile = [
            {
                "Function Name": func_name,
                "Total Time (s)": total_time,
                "Calls": self.function_calls[func_name],
                "Avg Time (s)": total_time / self.function_calls[func_name] if self.function_calls[func_name] > 0 else 0
            } for func_name, total_time in self.function_times.items()
        ]
        return {
            "functions": function_profile,
            "blocks": self.get_block_profile()
        }

    def print_line_profile(self):
        if profiler_config.enabled and profiler_config.print_line_profile:
            stats = self.line_profiler.get_stats()
            for (filename, start_lineno, func_name), timings in stats.timings.items():
                print(f"\nFile: {filename}")
                print(f"Function: {func_name}")
                print("Line #      Hits         Time (s)  Per Hit (s)   % Time  Line Contents")
                print("==============================================================")
                total_time = sum(timing[2] for timing in timings) / 1e6  # Convert to seconds
                for (lineno, nhits, time) in timings:
                    time_in_seconds = time / 1e6  # Convert to seconds
                    per_hit_in_seconds = time_in_seconds / nhits if nhits > 0 else 0
                    percent = (time_in_seconds / total_time) * 100 if total_time > 0 else 0
                    line_contents = linecache.getline(filename, lineno).rstrip()
                    print(f"{lineno:8d} {nhits:9d} {time_in_seconds:13.6f} {per_hit_in_seconds:11.6f} {percent:8.1f}  {line_contents}")

    def print_block_profile(self):
        if profiler_config.enabled and profiler_config.print_block_profile:
            block_profile = self.get_block_profile()
            print("\nBlock Profiling Report:")
            print(tabulate(block_profile, headers="keys", floatfmt=".6f", tablefmt="grid"))

    def print_overall_profile(self):
        if profiler_config.enabled and profiler_config.print_overall_profile:
            overall_profile = self.get_overall_profile()
            print("\nOverall Profiling Report:")
            print("Function Profiling:")
            print(tabulate(overall_profile['functions'], headers="keys", floatfmt=".6f", tablefmt="grid"))


profiler = LineProfilerBasedProfiler()
profile_methods = profiler.profile_methods
profile_line_by_line = profiler.profile_line_by_line
auto_profile_blocks = profiler.profile_function  # This is now equivalent to profile_function
