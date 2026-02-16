import pandas as pd
import ast
from datetime import datetime
import json

class CSVWorker:

    def csv_row_to_dict(csv_file, line_num):
        try:
            df = pd.read_csv(csv_file)
            line_num_str = line_num
            row = df.loc[df['line number'] == line_num_str]
            if row.empty:
                raise ValueError(f"Line number {line_num} (as string: '{line_num_str}') not found")
            return row.squeeze().to_dict()
        except FileNotFoundError:
            print(f"File {csv_file} not found")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


    def convert_data_types(params_dict):
        non_numeric_keys = ['id', 'calcium imaging directory', 'ephys directory',
                           'method_deconvolution', 'method_init', 'border_nan', 'LFP and EEG CSCs']
        converted_params = {}
        
        for key, value in params_dict.items():
            if key in non_numeric_keys:
                if key == 'LFP and EEG CSCs':
                    converted_params[key] = params_dict[key].split(";")
                    continue
                converted_params[key] = value
                continue
            
            converted_value = CSVWorker._convert_value(value, key)
            converted_params[key] = converted_value
    
        return converted_params

    def _convert_value(raw_value, key):
        """
        Converts a raw string value to its appropriate data type.
        """
        # NEW: Handle None
        if raw_value is None:
            return None

        # Check if the value is already a float
        if isinstance(raw_value, float):
            if pd.isna(raw_value): 
                return None
            else:
                return raw_value

        # Date conversion
        if key == 'date (YYMMDD)':
            return CSVWorker._convert_date(raw_value)

        # Ensure raw_value is a string for further processing
        if not isinstance(raw_value, str):
            return raw_value

        # NEW: Preprocess tuple-like strings for JSON
        processed_value = raw_value.strip().replace(" ", "")
        if processed_value.startswith("(") and processed_value.endswith(")"):
            processed_value = f'[{processed_value[1:-1]}]'

        # Attempt JSON parsing
        try:
            return json.loads(processed_value)
        except (json.JSONDecodeError, AttributeError) as e:
            pass

        # Attempt Python literal evaluation
        try:
            return ast.literal_eval(processed_value)
        except (ValueError, SyntaxError) as e:
            pass

        # Check for boolean strings
        lower_val = raw_value.lower()
        if lower_val == 'true':
            return True
        elif lower_val == 'false':
            return False

        # Check for None/empty
        elif lower_val == 'none' or raw_value.strip() == '':
            return None

        # Attempt float conversion
        try:
            return float(raw_value)
        except ValueError as e:
            return raw_value
        

    def _convert_date(date_str):
        """
        Converts a date string in the format YYMMDD to a datetime object.
        
        Args:
            date_str: The date string or float to be converted.
        
        Returns:
            datetime: The converted datetime object, or the original value if conversion fails.
        """
        try:
            if isinstance(date_str, float):
                date_str = str(int(date_str))
            if not isinstance(date_str, str):
                date_str = str(date_str)
            return datetime.strptime(date_str, '%y%m%d')
        except ValueError as e:
            return date_str