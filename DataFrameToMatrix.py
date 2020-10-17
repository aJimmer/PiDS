import pandas as pd
import numpy as np

class DataFrameToMatrix():
    """DataFrameToMatrix: Class that converts a DataFrame to a Numpy Matrix (ndarray)
        Notes:
            fit_transform: Does a fit and a transform and returns the transformed matrix
            transform: Based on previous fit parameters returns the transformed matrix
        More Info: https://supercowpowers.github.io/zat/dataframe_to_matrix.html
        # Nullable integer arrays are currently not handled by Numpy
        # Cast Nullable integer arrays to float32
        null_int_types = [pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype, pd.Int64Dtype]
        for col in _internal_df:
            if type(_internal_df[col].dtype) in null_int_types:
                _internal_df[col] = _internal_df[col].astype(np.float32)
    """

    def __init__(self):
        """Initialize the DataFrameToMatrix class"""
        self.column_names = None
        self.norm_map = {}
        self.dtype_map = {}
        self.nan_replace = -999

    def fit_transform(self, input_df, normalize=True, nan_replace=-999, copy=True):
        """Convert the dataframe to a matrix (numpy ndarray)
        Args:
            input_df (dataframe): The dataframe to convert
            normalize (bool): Boolean flag to normalize numeric columns (default=True)
        """
        self.nan_replace = nan_replace

        # Copy the dataframe (if wanted)
        _internal_df = input_df.copy() if copy else input_df

        # Convert object columns to categorical
        self.object_to_categorical(_internal_df)

        # Convert categorical NaNs
        self.fit_category_nans(_internal_df)

        # Lock categories to specific values (important for train/predict consistency)
        self.lock_categorical(_internal_df)

        # Sanity Check
        self.sanity_check_categorical(_internal_df)

        # Normalize numeric columns (mean normalize, sometimes called 'standardizing')
        if normalize:
            self.normalize_numeric(_internal_df)

        # Remove any numerical NaNs (categorical NaNs were removed above)
        for column in _internal_df.select_dtypes(include='number').columns:
            _internal_df[column].fillna(self.nan_replace, inplace=True)

        # Drop any columns that aren't numeric or categorical
        for column in list(_internal_df.select_dtypes(exclude=['number', 'category']).columns):
            print('Dropping {:s} column...'.format(column))
        _internal_df = _internal_df.select_dtypes(include=['number', 'category'])

        # Capture all the column/dtype information from the dataframe
        self.column_names = _internal_df.columns.to_list()
        for column in _internal_df.columns:
            self.dtype_map[column] = _internal_df[column].dtype

        # Now with every thing setup, call the dummy_encoder, convert to ndarray and return
        return pd.get_dummies(_internal_df).to_numpy(dtype=np.float32)
    
    @staticmethod
    def fit_category_nans(df):
        """ONLY FIT: Convert np.NaNs to a category 'NaN'"""
        for column in df.select_dtypes(include=['category']).columns:
            if df[column].isnull().any():
                df[column].cat.add_categories('NaN', inplace=True)
                df[column].fillna('NaN', inplace=True)

    @staticmethod
    def object_to_categorical(df):
        """Run a heuristic on the object columns to determine whether it contains categorical values
           if the heuristic decides it's categorical then the type of the column is changed
        Args:
            df (dataframe): The dataframe to check for categorical data
        Returns:
            None but the dataframe columns are modified
        """

        # Loop through each column that might be converable to categorical
        for column in df.select_dtypes(include='object').columns:

            # If we don't have too many unique values convert the column
            if df[column].nunique() < 100:
                print('Changing column {:s} to category...'.format(column))
                df[column] = pd.Categorical(df[column])
    
    @staticmethod
    def lock_categorical(df):
        """Lock the categorical column types to a specific ordered list of categories
        Args:
            df (dataframe): The dataframe to lock categorical columns
        Returns:
            None but note that the dataframe is modified to 'lock' the categorical columns
        """
        for column in df.select_dtypes(include='category').columns:
            df[column] = pd.Categorical(df[column], categories=sorted(df[column].unique().tolist()))
    
    @staticmethod
    def sanity_check_categorical(df):
        """Sanity check for 'dimensionality explosion' on categorical types
        Args:
            df (dataframe): The dataframe to check the categorical columns
        Returns:
            None
        """
        for column in df.select_dtypes(include='category').columns:
            # Give warning on category types will LOTs of values
            num_unique = df[column].nunique()
            if num_unique > 20:
                print('WARNING: {:s} will expand into {:d} dimensions...'.format(column, num_unique))

    def normalize_numeric(self, df):
        """Normalize (mean normalize) the numeric columns in the dataframe
        Args:
            df (dataframe): The dataframe to normalize
        Returns:
            None but note that the numeric columns of the dataframe are modified
        """
        for column in df.select_dtypes(include='number').columns:
            print('Normalizing column {:s}...'.format(column))
            df[column] = self._normalize_series(df[column])
    
    def _normalize_series(self, series):
        smin = series.min()
        smax = series.max()

        # Check for div by 0
        if smax - smin == 0:
            print('Cannot normalize series (div by 0) so not normalizing...')
            return series

        # Capture the normalization info and return the normalize series
        self.norm_map[series.name] = (smin, smax)
        return (series - smin) / (smax - smin)