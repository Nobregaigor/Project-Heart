from project_heart.enums import SCRIPT_TAGS
from .utils import read_json
import logging
import os
from pathlib import Path

logger = logging.getLogger(name="ScriptHandler")
logging.basicConfig()
logger.setLevel(logging.INFO)

class ScriptHandler():
    
    @staticmethod
    def get_json_data(filepath):
        assert filepath.endswith(".json"), "File path must end with .json to be correctly parsed as JSON"
        return read_json(filepath)

    @staticmethod
    def resolve_json_input_format(**kwargs):
        logger.debug("resolving input format.")
        # Get input data
        # --> either provided through 'json_file' or kwargs
        file_input = kwargs.get(SCRIPT_TAGS.JSON.value, None)
        # check if file_input was provided
        if file_input is not None: # if user did not specify output
            logger.info("Found json file as input: '{}'".format(file_input))
            input_data = ScriptHandler.get_json_data(file_input)
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
            logger.info("Input data is in recursive mode: '{}'".format(type(input_data)))
            for data in input_data:
                fun(**data)
        
    @staticmethod
    def resolve_multiple_input_files(input_data, fun):

        # check if provided function is valid
        if not callable(fun):
            raise ValueError("Input function must be callable.")
        
        # get input_dir tag
        in_dir = input_data.get(SCRIPT_TAGS.INPUT_DIR.value, None)
        if in_dir is not None:
            logger.info("Found input directory: '{}'".format(in_dir))
            # ensure in_dir is Path object
            in_dir = Path(in_dir)
            # check if use specified extension.
            ext = input_data.get(SCRIPT_TAGS.INPUT_EXT.value, None)
            if ext is None:
                logger.info("No extension specified for input directory. "
                "ScriptHandler will attempt to use all files in directory. "
                "If any files in '{}' do not match specified function requirements,"
                "error might occur. To prevent failure due to file extension, "
                "either specify extension with 'input_ext': '.ext' or ensure "
                "all files in directory have the same extension."
                .format(str(in_dir)))

                all_files = [in_dir/f for f in os.listdir(str(in_dir))]
            else:
                logger.info("Found extension: '{}'".format(ext))
                if not isinstance(ext, str):
                    raise ValueError("Invalid extension. Must be a string. Example: '.txt'")

                all_files = [in_dir/f for f in os.listdir(str(in_dir)) if f.endswith(ext)]

            logger.info("Total number of files: {}".format(len(all_files)))
            # Ensure SCRIPT_TAGS.INPUT_DIR is None (so function do not repeat itself)
            input_data[SCRIPT_TAGS.INPUT_DIR.value] = None
            for file in all_files:
                input_data[SCRIPT_TAGS.INPUT_FILE.value] = file
                logger.debug("Input args for file: {}".format(input_data))
                fun(**input_data)
            
    
    @staticmethod
    def resolve_multiple_input_arguments(input_data, fun):
        pass
    
    @staticmethod
    def add_filename_data_to_df(df, filename, namemap):
        # add data from filename is requested
        contents = os.path.basename(filename).split("_") #split filename based on "_"
        for ik, key in enumerate(namemap):
            if len(key) > 0: # user can skip content if key is "" (no length)
                if key.startswith("neg_float_"):  # this option allows user to add content as a negative float
                    key = key.replace("neg_float_", "")
                    df[key] = -float(contents[ik])
                elif key.startswith("float_"): # this option allows user to add content as a float
                    key = key.replace("float_", "")
                    df[key] = float(contents[ik])
                else: 
                    df[key] = contents[ik]
        return df

    @staticmethod
    def merge_df_with_existing_at_file(df, outpath, **kwargs):
        import pandas as pd
        if os.path.exists(outpath):
            try:
                if outpath.endswith('.csv'):
                    other_df = pd.read_csv(outpath, **kwargs)
                elif outpath.endswith('.ftr'):
                    other_df = pd.read_feather(outpath, **kwargs)
                elif outpath.endswith('.xml'):
                    other_df = pd.read_xml(outpath, **kwargs)
                else:
                    raise RuntimeError("Unable to read {}"
                    "as pandas dataframe. We current do not "
                    "support this format. Options are: "
                    ".csv, .ftr, .xml".format(outpath))
                return pd.concat((df, other_df))
            except:
                raise RuntimeError("Unable to read {}"
                    "as pandas dataframe. Please, disable "
                    "'merge_df_with_existing_at_file' or specify " 
                    "a different output path.".format(outpath))
        return df

    @staticmethod
    def export_df(df, outpath, **kwargs):
        if outpath.endswith('.csv'):
            df.to_csv(outpath, **kwargs)
        elif outpath.endswith('.ftr'):
            if "index" in kwargs:
                from copy import copy
                use_args = copy(kwargs)
                use_args.pop("index")
            else:
                use_args = kwargs
            df = df.reset_index()
            df.to_feather(outpath, **use_args)
        elif outpath.endswith('.xml'):
            df.to_xml(outpath, **kwargs)
        else:
            raise ValueError("Unknown output format. "
            "Maybe you want to export to a different format "
            "than those we support in this function? "
            "Try exporting 'df' manually. "
            "Check https://pandas.pydata.org/docs/user_guide/io.html"
            )



    @staticmethod
    def assert_input_file(input_file, ext=None):
        assert input_file is not None, "Input file must be specified. Received: {}".format(input_file)
        assert os.path.isfile(input_file), "Input file is not a file. Please, verify: {}".format(input_file)
        if ext is not None:
            assert input_file.endswith(ext), "Input file does not have the extension format '{}'. Received: {}".format(ext, input_file)
    
    @staticmethod
    def assert_input_exists(arg, match_type=None):
        assert arg is not None, "{} argument must be specified.".format(arg)
        if match_type is not None:
            assert isinstance(arg, match_type), "{} argument must be of type: {}".format(arg, match_type)
    
    
    @staticmethod
    def assert_filename_data(filename, namemap):
        contents = os.path.basename(filename).split("_") #split filename based on "_"
        assert len(contents) == len(namemap), (
            "Number of contents in a filename map must match exactly with number of filename contents. "
            "Contents are defined as string values split by '_'. \n"
            "For example: test_1_may.csv -> ['run', 'id', 'month']. \n"
            "If you would like to skip content, you can use an empty string ''. \n"
            "For example: test_1_irrelevant_may.csv -> ['run', 'id', '', 'month']. \n"
            "Expected: {}. Received: {}".format(len(contents), len(namemap))
        )
