from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ProfilerConfig:
    enabled: bool = True
    output_dir: str = "profiling_output"
    save_format: str = "json"  # Can be 'json' or 'txt'
    steps_to_save: List[int] = field(default_factory=lambda: [1, 100, 1000])
    
    # Printing options
    print_line_profile: bool = False
    print_block_profile: bool = True
    print_overall_profile: bool = True
    print_save_info: bool = True
    
    # Saving options
    save_line_profile: bool = True
    save_block_profile: bool = True
    save_overall_profile: bool = True

    # Advanced options
    line_by_line_file: Optional[str] = None
    block_profiling_file: Optional[str] = None
    overall_profiling_file: Optional[str] = None

profiler_config = ProfilerConfig()