from scipy.io import arff
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, normalize,\
                                    MinMaxScaler
from sklearn.impute import SimpleImputer

import config
from src.preprocessing.preprocessing import col_standardisation,\
                                            imputate_row_missing_values_with_median,\
                                            imputate_row_missing_values_with_mode,\
                                            df_to_string, df_to_int, df_to_numeric,\
                                            one_hot_encoding, label_encoding
class Processing:
    """
    Class to ingest and process data for clustering algorithms.
    """

    def __init__(self, 
                 source_path=config.SOURCE_PATH, 
                 name=None, 
                 df=None, 
                 d_features=config.D_FEATURES,
                 thresh_cat=config.THRESH_CAT):
        
        self.source_path = source_path
        self.name = name
        self.df = df
        self.d_features = d_features
        self.thresh_cat = thresh_cat

    def read_info(self, name):
        df_info = pd.read_csv(os.path.join(self.source_path, "info.tsv"), sep="\t")

        df_info["Name"] = (
            df_info["Name"]
            .str.replace("*", "")
            .str.strip()
            .str.replace(" ", "_")
            .str.lower()
        )

        df_info.set_index("Name", inplace=True, drop=True)
        df_info = df_info.loc[name, :]

        return df_info

    def read(self, name: str, metadata: bool = False):
        """
        Reads arff files and generates corresponding pandas DataFrame with data.

        Args:
            name: string with the name of the file to read
            path: path to the file to read
            metadata: Boolean to indicate if the metadata is printed
        
        Returns:
            df: DataFrame containing dataset data.
        """
        self.name = name

        with open(os.path.join(self.source_path, f"{name}.arff"), "r") as file:
            data, meta = arff.loadarff(file)

        # Convert data to a pandas DataFrame
        df = pd.DataFrame(data)

        # Strings are read as bytes, so we need to decode them
        for column in df.columns:
            if df[column].dtype == "object":
                df[column] = df[column].str.decode("utf-8")

        logging.info(f"\nData read successfully ({df.shape[0]} rows, {df.shape[1]} columns)")

        if metadata:
            # Get info about the dataset
            logging.info("\nInfo about the dataset:")
            logging.info(meta)

        self.df = df
        return df
    
    def general_preprocessing(self):
        '''
        General Preprocessor

        Args:
            df: DataFrame containing dataset data.
            name: string with the name of the dataset
            d_features: dictionary with the features of each dataset

        Preprocess:
            Numeric columns:
                - Convert to float
                - Imputate missing values with median
                - Scale data
            Categoric columns:
                - Convert to string
                - Imputate missing values with mode
                - One hot encode if less than THRESH_CAT categories
                - Label encode if more than THRESH_CAT categories
            Categoric ordinal columns:
                If string categories:
                    - Convert to string
                    - Map values to integers
                    - Convert to int
                If already int categories:
                    - Convert to int
        
        '''

        # Get the dictionary from config
        d_features = self.d_features[self.name]

        prep_df = self.df.copy()
        columns = prep_df.columns

        # Extract features by type
        num_col = d_features.get("numeric", [])
        cat_col = d_features.get("nominal", [])
        cat_ord = d_features.get("nominal_ordinal", {})

        num_col = self.check_features_in_df(prep_df, num_col)
        cat_col = self.check_features_in_df(prep_df, cat_col)
        ls_cat_ord = self.check_features_in_df(prep_df, list(cat_ord.keys()))
        cat_ord = {key: value for key, value in cat_ord.items() if key in columns}

        prep_df = prep_df.loc[:, num_col + cat_col + ls_cat_ord]

        # Numeric preprocessing
        if num_col != []:

            # Ensure the values arre Real numbers
            prep_df = df_to_numeric(prep_df, num_col)

            # Impute Missing Values
            prep_df = imputate_row_missing_values_with_median(prep_df, num_col)

            # Scale data
            prep_df = col_standardisation(prep_df, num_col)

        # Categoric (Nominal) preprocessing
        if cat_col != []:

            # Convert to string
            prep_df = df_to_string(prep_df, cat_col)

            # Impute with mode
            prep_df = imputate_row_missing_values_with_mode(prep_df, cat_col)

            # If the number of categories is too high, OHE encoding creates too many
            # columns. This leads to Dimensionality Curse. Therefore, we use Label Encoding
            # Using Label Encoding is not the best option because we stablish a non-existent 
            # order, but it is better than OHE

            ls_ohe = [] # List of columns to One Hot Encode
            ls_lab = [] # List of columns to Label Encode

            for col in cat_col:

                if prep_df[col].nunique() <= self.thresh_cat:
                    ls_ohe.append(col)
                else:
                    ls_lab.append(col)

            # Perform the encodings
            if ls_ohe != []:
                prep_df = one_hot_encoding(prep_df, ls_ohe)
            
            if ls_lab != []:
                prep_df = label_encoding(prep_df, ls_lab)

        # Categoric (Ordinal) preprocessing
        # In this case there are two options:
        #   - The categories are strings and we must stablish an order
        #   - The categories are already ordered integers
        if cat_ord != {}:
            # In this case, we have a dictionary with the columns to be encoded and
            # the order of the categories
            cat_ord_cols = list(cat_ord.keys()) # Nominal Ordinal columns

            # Manage the abovementioned cases
            for col in cat_ord_cols:
                if cat_ord[col] != {}:
                    # Map values to integers
                    prep_df = df_to_string(prep_df, [col])
                    prep_df.loc[:, col] = prep_df.loc[:, col].map(cat_ord[col])
                else:
                    # Convert to int
                    prep_df = df_to_int(prep_df, [col])
        
        self.df = prep_df
        
        return prep_df
    
    @staticmethod
    def check_features_in_df(df, cols):
        columns = df.columns
        # Manage numeric columns not found in the dataset
        out_cols = [col for col in cols if col not in columns]
        if len(out_cols) > 0:

            logging.warning(f"Columns {out_cols} are not in the dataset")
            cols = [col for col in cols if col in columns]

            if cols == []:
                logging.warning(f"None of the ordinal columns {cols} have been detected")
        return cols
    

if __name__ == "__main__":
     import numpy as np

     dataclass = Processing(source_path='input/datasets/')
     df = dataclass.read('glass')
     df = df.iloc[:, :-1]

     print(df.loc[0, "Mg"])
     df.loc[0, "Mg"] = np.nan
     print(df.loc[0, "Mg"])
     df = dataclass.general_preprocessing()
     print(df.loc[0, "Mg"])

     print(df["Na"].mean(), df["Mg"].mean())

     print(df.head())