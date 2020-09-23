
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