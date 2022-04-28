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

    MESH_TYPE = "mesh_type"

    PREFIX_MAP = "prefix_map" # adds prefix to the filename (must be list)

    SAVE_VTK = "save_vtk"


class LV_SCRIPT_TAGS(Enum):
    LOG_LEVEL = "lv_log_level"
    IDENTIFY_REGIONS = "identify_regions"
    SPECKLES = "speckles"
    METRICS = "metrics"
    
    # params used for fiber computation
    ALPHA_ENDO = "alpha_endo" # endo angle (must be positive)
    ALPHA_EPI = "alpha_epi" # epi angle (must be negative)
    FIBERS = "fibers" # kwargs for fiber ldrb
    TETRAHEDRALIZE = "tetrahedralize" #kwargs for tetrahedralization of non-tetra meshes
    
    REGRESS = "regress" #kwargs for regression (from tetra to non-tetra mesh)
    INTERPOLATE = "interpolate" #kwargs for interpolation (from tetra to non-tetra mesh)

    FEB_TEMPLATE = "feb_template"

    BOUNDARY_CONDITIONS = "boundary_conditions"
    



    