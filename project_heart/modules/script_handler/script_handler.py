from project_heart.enums import SCRIPT_TAGS
from .utils import read_json
import logging

logger = logging.getLogger(name="ScriptHandler")

class ScriptHandler():
    
    @staticmethod
    def resolve_json_input_format(**kwargs):
        
        # Get input data
        # --> either provided through 'json_file' or kwargs
        file_input = kwargs.get(SCRIPT_TAGS.JSON.value, None)
        # check if file_input was provided
        if file_input is not None: # if user did not specify output
            input_data = read_json(file_input)
        else:
            input_data = kwargs
        # validade input arguments (must be greater than zero)
        if len(input_data) == 0:
            raise ValueError("No input data provided: {}".format(input_data))
        # return data as a dictionary
        return input_data
    
    @staticmethod
    def resolve_recursive(input_data, fun):
        # Decide wheter to enter 'recursive' mode
        
        # check if provided function is valid
        if not callable(fun):
            raise ValueError("Input function must be callable.")

        # If input is a list of data, therefore multiple instance to compute 
        # fiber directions separately, recursevely apply each element
        if isinstance(input_data, (list, tuple)):
            for data in input_data:
                fun(**data)
            return
        
    @staticmethod
    def resolve_multiple_input_files(input_data, fun):

        # check if provided function is valid
        if not callable(fun):
            raise ValueError("Input function must be callable.")
        
        # get input_dir tag
        in_dir = kwargs.get(SCRIPT_TAGS.INPUT_DIR, None)
        if in_dir is not None:
            
            # ensure in_dir is Path object
            in_dir = pathlib.Path(in_dir)
            # check if use specified extension.
            ext = kwargs.get(SCRIPT_TAGS.INPUT_EXT, None)
            if ext is None:
                logger.info("No extension specified for input directory. "
                "ScriptHandler will attempt to use all files in directory. "
                "If any files in '{}' do not match specified function requirements,"
                "error might occur. To prevent failure due to file extension, "
                "either specify extension with 'input_ext': '.ext' or ensure "
                "all files in directory have the same extension."
                .format(str(in_dir)))

                all_files = [in_dir/f for f in listdir(str(in_dir))]
            else:
                if not isinstance(ext, str):
                    raise ValueError("Invalid extension. Must be a string. Example: '.txt'")

                all_files = [in_dir/f for f in listdir(str(in_dir)) if f.endswith(ext)]

            # Ensure SCRIPT_TAGS.INPUT_DIR is None (so function do not repeat itself)
            input_data[SCRIPT_TAGS.INPUT_DIR] = None
            for file in all_files:
                input_data[SCRIPT_TAGS.INPUT_FILE] = file
                fun(**input_data)
                return
    
    @staticmethod
    def resolve_multiple_input_arguments(input_data, fun):
        pass