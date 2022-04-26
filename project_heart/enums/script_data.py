from enum import Enum

class SCRIPT_TAGS(Enum):
    JSON = "json_file"

    INPUT_FILE = "input_file"
    OUTPUT_FILE = "output_file"
    
    INPUT_DIR = "input_directory"
    INPUT_EXT = "input_ext"

    LOG_LEVEL = "log_level"
    DTYPE = "dtype"

    FILENAME_MAP = "filename_map"

    MERGE_WITH_EXISTING_FILE = "merge_with_existing_file"


class LV_SCRIPT_TAGS(Enum):
    IDENTIFY_REGIONS = "identify_regions"
    SPECKLES = "speckles"
    METRICS = "metrics"
    
    