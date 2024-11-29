import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Define a configuration dataclass to hold the file path for saving the preprocessor object.
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")

# DataTransformation class handles data preprocessing.
class DataTransformation:
    def __init__(self):
        # Initialize the data transformation configuration.
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for creating the preprocessing object, 
        which includes transformations for both numerical and categorical columns.
        '''
        try:
            # Define the numerical columns to be transformed.
            numerical_columns = ["writing_score", "reading_score"]
            # Define the categorical columns to be transformed.
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical data: handles missing values and scales the data.
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing values with the median.
                    ("scaler", StandardScaler())                   # Standardize the data.
                ]
            )

            # Pipeline for categorical data: handles missing values, applies one-hot encoding, and scales the data.
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing values with the most frequent value.
                    ("one_hot_encoder", OneHotEncoder()),                 # Convert categorical data to one-hot encoded format.
                    ("scaler", StandardScaler(with_mean=False))           # Scale the encoded data.
                ]
            )

            # Log the columns being transformed.
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine the numerical and categorical pipelines into a single transformer.
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns.
                    ("cat_pipelines", cat_pipeline, categorical_columns) # Apply cat_pipeline to categorical columns.
                ]
            )

            # Return the preprocessor object.
            return preprocessor
        
        except Exception as e:
            # Raise a custom exception if an error occurs.
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        This function handles the application of the preprocessing object to the training and testing datasets.
        '''
        try:
            # Read the training and testing datasets from the specified paths.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            # Obtain the preprocessing object.
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            # Define the target column name.
            target_column_name = "math_score"
            # Define numerical columns to retain consistent reference.
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target feature from the training dataset.
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target feature from the testing dataset.
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Apply the preprocessing object to the training and testing input features.
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine the processed input features with the target feature for both training and testing datasets.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            # Save the preprocessing object to the specified file path.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the transformed training and testing arrays along with the file path of the preprocessor object.
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            # Raise a custom exception if an error occurs.
            raise CustomException(e, sys)
