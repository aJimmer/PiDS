from ZeekLogReader import ZeekLogReader
import pandas as pd

class LogToDataFrame(object):
    """LogToDataFrame: Converts a Zeek log to a Pandas DataFrame
        Notes:
            This class has recently been overhauled from a simple loader to a more
            complex class that should in theory:
              - Select better types for each column
              - Should be faster
              - Produce smaller memory footprint dataframes
            If you have any issues/problems with this class please submit a GitHub issue.
        More Info: https://supercowpowers.github.io/zat/large_dataframes.html
    """
    def __init__(self):
        """Initialize the LogToDataFrame class"""

        # First Level Type Mapping
        #    This map defines the types used when first reading in the Zeek log into a 'chunk' dataframes.
        #    Types (like time and interval) will be defined as one type at first but then
        #    will undergo further processing to produce correct types with correct values.
        # See: https://stackoverflow.com/questions/29245848/what-are-all-the-dtypes-that-pandas-recognizes
        #      for more info on supported types.
        self.type_map = {'bool': 'category',  # Can't hold NaN values in 'bool', so we're going to use category
                         'count': 'UInt64',
                         'int': 'Int32',
                         'double': 'float',
                         'time': 'float',      # Secondary Processing into datetime
                         'interval': 'float',  # Secondary processing into timedelta
                         'port': 'UInt16'
                         }

    def _get_field_info(self, log_filename):
        """Internal Method: Use ZAT log reader to read header for names and types"""
        _zeek_reader = ZeekLogReader(log_filename)
        _, field_names, field_types, y = _zeek_reader._parse_zeek_header(log_filename)
        print('offset: '+ str(_) + '\n') 
        print('field names: ') 
        print(field_names) 
        print('\nfield_types: ') 
        print(field_types)  
        print('\ntype_converters: ' + str(y))
        
        return field_names, field_types

    def _create_initial_df(self, log_filename, all_fields, usecols, dtypes):
        """Internal Method: Create the initial dataframes by using Pandas read CSV (primary types correct)"""
        return pd.read_csv(log_filename, sep='\t', names=all_fields, usecols=usecols, dtype=dtypes, comment="#", na_values='-')

    def create_dataframe(self, log_filename, ts_index=True, aggressive_category=True, usecols=None):
        """ Create a Pandas dataframe from a Bro/Zeek log file
            Args:
               log_fllename (string): The full path to the Zeek log
               ts_index (bool): Set the index to the 'ts' field (default = True)
               aggressive_category (bool): convert unknown columns to category (default = True)
               usecol (list): A subset of columns to read in (minimizes memory usage) (default = None)
        """

        # Grab the field information
        field_names, field_types = self._get_field_info(log_filename)
        all_fields = field_names  # We need ALL the fields for later

        # If usecols is set then we'll subset the fields and types
        if usecols:
            # Usecols needs to include ts
            if 'ts' not in usecols:
                usecols.append('ts')
            field_types = [t for t, field in zip(field_types, field_names) if field in usecols]
            field_names = [field for field in field_names if field in usecols]

        # Get the appropriate types for the Pandas Dataframe
        pandas_types = self.pd_column_types(field_names, field_types, aggressive_category)

        # Now actually read in the initial dataframe
        self._df = self._create_initial_df(log_filename, all_fields, usecols, pandas_types)

        # Now we convert 'time' and 'interval' fields to datetime and timedelta respectively
        for name, zeek_type in zip(field_names, field_types):
            if zeek_type == 'time':
                self._df[name] = pd.to_datetime(self._df[name], unit='s')
            if zeek_type == 'interval':
                self._df[name] = pd.to_timedelta(self._df[name], unit='s')

        # Set the index
        if ts_index and not self._df.empty:
            self._df.set_index('ts', inplace=True)
        return self._df

    def pd_column_types(self, column_names, column_types, aggressive_category=True, verbose=False):
        """Given a set of names and types, construct a dictionary to be used
           as the Pandas read_csv dtypes argument"""

        # Agressive Category means that types not in the current type_map are
        # mapped to a 'category' if aggressive_category is False then they
        # are mapped to an 'object' type
        unknown_type = 'category' if aggressive_category else 'object'

        pandas_types = {}
        for name, zeek_type in zip(column_names, column_types):

            # Grab the type
            item_type = self.type_map.get(zeek_type)

            # Sanity Check
            if not item_type:
                # UID/FUID/GUID always gets mapped to object
                if 'uid' in name:
                    item_type = 'object'
                else:
                    if verbose:
                        print('Could not find type for {:s} using {:s}...'.format(zeek_type, unknown_type))
                    item_type = unknown_type

            # Set the pandas type
            pandas_types[name] = item_type

        # Return the dictionary of name: type
        return pandas_types
