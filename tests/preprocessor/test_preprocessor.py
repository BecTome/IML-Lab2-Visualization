import pytest
import pandas as pd
import numpy as np
from src.read.processing import Processing

# Create dataframe with 
#  2 numeric columns 
#  2 categorical columns (one of them less than THRESH_CAT categories)
#  2 categorical ordinal columns (one of them with string categories)

df = pd.DataFrame({"numeric_col1": [103.5, 290.34, 155.7, 29.0],
                   "numeric_col2": [0.5, 0.34, 0.7, 0.0],
                   "cat_col1": ["cat1", "cat2", "cat1", "cat1"],
                   "cat_col2": ["cat1", "cat2", "cat3", "cat4"],
                   "cat_ord_col1": ["low", "medium", "high", "medium"],
                   "cat_ord_col2": [0, 1, 2, 1]})

df_out = pd.DataFrame({
                        'numeric_col1': [-0.431118, 1.527069, 0.115967, -1.211919],
                        'numeric_col2': [0.448743, -0.175595, 1.229166, -1.502314],
                        'cat_col2': [0, 1, 2, 3],
                        'cat_ord_col1': [0, 1, 2, 1],
                        'cat_ord_col2': [0, 1, 2, 1],
                        'cat_col1_cat2': [0, 1, 0, 0]
                    })

d_features = {"test": {
              "numeric":["numeric_col1", "numeric_col2"],
              "nominal":["cat_col1", "cat_col2"],
              "nominal_ordinal": {"cat_ord_col1": {"low": 0, "medium":1, "high":2},
                                  "cat_ord_col2": {}}}
                }
thresh_cat = 2

processor = Processing(df=df, name="test", d_features=d_features, thresh_cat=thresh_cat)
processor.general_preprocessing()

def test_numeric():
    assert (df_out["numeric_col1"] - processor.df["numeric_col1"]).abs().max() < 1e-3 ,\
           "Numeric column 1 is not equal"
    assert (df_out["numeric_col2"] - processor.df["numeric_col2"]).abs().max() < 1e-3,\
           "Numeric column 2 is not equal"

def test_categorical_under_threshold():
    assert (df_out["cat_col1_cat2"] - processor.df["cat_col1_cat2"]).abs().max() < 1e-3,\
           "Categorical column 1 is not equal"

def test_categorical_over_threshold():
    assert (df_out["cat_col2"] - processor.df["cat_col2"]).abs().max() < 1e-3,\
           "Categorical column 1 is not equal"

def test_categorical_ordinal_string():
    assert df_out["cat_ord_col1"].equals(processor.df["cat_ord_col1"]),\
          "Categorical ordinal column 1 is not equal"

def test_categorical_ordinal_int():
    assert df_out["cat_ord_col2"].equals(processor.df["cat_ord_col2"]),\
          "Categorical ordinal column 2 is not equal"





