from dataclasses import dataclass 
from pathlib import Path 


@dataclass
class DataIngestionconfig:
    root_dir: Path
    source_URL: str 
    local_data_file: Path 
    unzip_dir: Path 

@dataclass
class DataValidationconfig:
    root_dir: Path
    unzip_data_dir: Path
    STATUS_FILE: Path
    all_schema: dict 
