import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:
    '''
        Desc: This class will initialize path for artifact folder with .csv file under that
    '''
    train_data_path = os.path.join('artifact','train.csv')
    test_data_path = os.path.join('artifact','test.csv')
    raw_data_path = os.path.join('artifact','raw.csv')
    

class DataIngestion:
    '''
        Desc: This class will read data and split it into train test and than
              store it in our raw train test data path
    '''
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info('Entered in data ingestion')
            data = pd.read_csv(os.path.join("data","data.csv"))
            logging.info('Data read as data frame')
            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path),exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path,index=False)
            logging.info('Raw data saved successfully')
            logging.info('Data splitted into train test')
            train_set, test_set = train_test_split(data,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)
            logging.info('Splitted data saved successfully')
            logging.info('Data ingestion completed')
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
    
        


