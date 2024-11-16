import os
import sys
import pandas as pd
from src.exception import CustomException  # Custom exception handling class
from src.logger import logging  # Custom logging utility
from sklearn.model_selection import train_test_split  # For splitting the dataset
from dataclasses import dataclass  # For creating configuration classes easily

# Dataclass to store file paths for data ingestion
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to save the train dataset
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to save the test dataset
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')  # Path to save the raw dataset

# Class to handle the data ingestion process
class DataIngestion:
    def __init__(self):
        # Initialize the ingestion configuration with default paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Reads raw data, splits it into training and testing datasets, and saves them.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the raw dataset
            df = pd.read_csv('notebooks\data\stud.csv')  # Path to the input dataset
            logging.info("Read the dataset as dataframe")

            # Create the directory for saving artifacts if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw dataset to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test Split initiated")
            
            # Split the dataset into training and testing sets (80-20 split)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=39)

            # Save the train dataset to the specified path
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            # Save the test dataset to the specified path
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")

            # Return the paths of the train and test datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Raise a custom exception if an error occurs
            raise CustomException(e, sys)

# Entry point for the script
if __name__ == "__main__":
    # Create an instance of DataIngestion
    obj = DataIngestion()
    # Initiate the data ingestion process
    obj.initiate_data_ingestion()
