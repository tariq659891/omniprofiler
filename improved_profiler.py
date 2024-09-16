import time
from functools import wraps
from collections import defaultdict
import json
import os
from contextlib import contextmanager
from profiler_config import profiler_config
import line_profiler
import io

class AdvancedFlexibleProfiler:
    def __init__(self):
        self.function_times = defaultdict(float)
        self.function_calls = defaultdict(int)
        self.block_times = defaultdict(float)
        self.block_calls = defaultdict(int)
        self.line_profiler = line_profiler.LineProfiler()
        self.current_block = None
        self.block_start_time = None
        self.step = 0

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
            result = func(*args, **kwargs)
            end_time = time.time()
            self.function_times[func.__name__] += end_time - start_time
            self.function_calls[func.__name__] += 1
            return result
        return wrapper

    def auto_profile_blocks(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler_config.enabled:
                return func(*args, **kwargs)
            self.start_block(func.__name__)
            result = func(*args, **kwargs)
            self.end_block()
            return result
        return wrapper

    def start_block(self, name):
        if self.current_block:
            self.end_block()
        self.current_block = name
        self.block_start_time = time.time()

    def end_block(self):
        if self.current_block:
            end_time = time.time()
            self.block_times[self.current_block] += end_time - self.block_start_time
            self.block_calls[self.current_block] += 1
            self.current_block = None
            self.block_start_time = None

    def profile_line_by_line(self, func):
        if not profiler_config.enabled:
            return func
        return self.line_profiler(func)

    def get_line_profile(self):
        stream = io.StringIO()
        self.line_profiler.print_stats(stream=stream)
        return stream.getvalue()

    def get_block_profile(self):
        return {
            block_name: {
                "total_time": total_time,
                "calls": self.block_calls[block_name],
                "avg_time": total_time / self.block_calls[block_name] if self.block_calls[block_name] > 0 else 0
            } for block_name, total_time in self.block_times.items()
        }

    def get_overall_profile(self):
        return {
            "functions": {
                func_name: {
                    "total_time": total_time,
                    "calls": self.function_calls[func_name],
                    "avg_time": total_time / self.function_calls[func_name] if self.function_calls[func_name] > 0 else 0
                } for func_name, total_time in self.function_times.items()
            },
            "blocks": self.get_block_profile()
        }

    def print_line_profile(self):
        if profiler_config.enabled and profiler_config.print_line_profile:
            print(self.get_line_profile())

    def print_block_profile(self):
        if profiler_config.enabled and profiler_config.print_block_profile:
            block_profile = self.get_block_profile()
            print("\nBlock Profiling Report:")
            for block_name, data in block_profile.items():
                print(f"{block_name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s")

    def print_overall_profile(self):
        if profiler_config.enabled and profiler_config.print_overall_profile:
            overall_profile = self.get_overall_profile()
            print("\nOverall Profiling Report:")
            print("Function Profiling:")
            for func_name, data in overall_profile['functions'].items():
                print(f"{func_name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s")
            print("\nBlock Profiling:")
            for block_name, data in overall_profile['blocks'].items():
                print(f"{block_name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s")

    def save_line_profile(self, step=None, filename=None):
        if profiler_config.enabled and profiler_config.save_line_profile:
            os.makedirs(profiler_config.output_dir, exist_ok=True)
            if filename is None:
                filename = os.path.join(profiler_config.output_dir, f"line_profile_step_{step or 'custom'}.txt")
            with open(filename, 'w') as f:
                f.write(self.get_line_profile())
            if profiler_config.print_save_info:
                print(f"Saved line-by-line profile to {filename}")

    def save_block_profile(self, step=None, filename=None):
        if profiler_config.enabled and profiler_config.save_block_profile:
            os.makedirs(profiler_config.output_dir, exist_ok=True)
            if filename is None:
                filename = os.path.join(profiler_config.output_dir, f"block_profile_step_{step or 'custom'}.{profiler_config.save_format}")
            self._save_report(filename, self.get_block_profile())
            if profiler_config.print_save_info:
                print(f"Saved block profile to {filename}")

    def save_overall_profile(self, step=None, filename=None):
        if profiler_config.enabled and profiler_config.save_overall_profile:
            os.makedirs(profiler_config.output_dir, exist_ok=True)
            if filename is None:
                filename = os.path.join(profiler_config.output_dir, f"overall_profile_step_{step or 'custom'}.{profiler_config.save_format}")
            self._save_report(filename, self.get_overall_profile())
            if profiler_config.print_save_info:
                print(f"Saved overall profile to {filename}")

    def _save_report(self, filename, report):
        if profiler_config.save_format == 'json':
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            with open(filename, 'w') as f:
                self._write_report_txt(f, report)

    def _write_report_txt(self, file, report):
        if isinstance(report, dict) and 'functions' in report:
            file.write("Function Profiling Report:\n")
            for func_name, data in report['functions'].items():
                file.write(f"{func_name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s\n")
            file.write("\nBlock Profiling Report:\n")
            for block_name, data in report['blocks'].items():
                file.write(f"{block_name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s\n")
        else:
            for name, data in report.items():
                file.write(f"{name}: Total time: {data['total_time']:.6f}s, Calls: {data['calls']}, Avg time: {data['avg_time']:.6f}s\n")

    def step_profiler(self):
        if not profiler_config.enabled:
            return

        self.step += 1
        if self.step in profiler_config.steps_to_save:
            self.save_overall_profile(self.step)
            self.save_line_profile(self.step)
            self.save_block_profile(self.step)
            self.print_overall_profile()
            self.print_line_profile()
            self.print_block_profile()

    def reset(self):
        self.function_times.clear()
        self.function_calls.clear()
        self.block_times.clear()
        self.block_calls.clear()
        self.line_profiler.clear()

    @contextmanager
    def profile_context(self, name):
        if not profiler_config.enabled:
            yield
            return

        self.start_block(name)
        try:
            yield
        finally:
            self.end_block()

profiler = AdvancedFlexibleProfiler()
profile_methods = profiler.profile_methods
profile_line_by_line = profiler.profile_line_by_line
auto_profile_blocks = profiler.auto_profile_blocks