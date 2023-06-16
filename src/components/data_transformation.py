import os
import sys
import pandas as pd 
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

@dataclass
class DataTransformationConfig:
    '''
        Desc: This class will create folder under artifact folder and save preprocessor pickle
    '''
    preprocessor_obj_file_path = os.path.join('artifact/data_transformation','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):
        '''
            Desc: This function will transform our data fill missing value and scaling data
        '''
        try:
            logging.info('Data transformation started')
            df = pd.read_csv(os.path.join("artifact/data_ingestion","raw.csv"))
            numerical_columns = ['mean_radius',
                                'mean_texture',
                                'mean_smoothness',
                                'mean_compactness',
                                'mean_symmetry',
                                'mean_fractal_dimension',
                                'radius_error',
                                'texture_error',
                                'smoothness_error',
                                'compactness_error',
                                'concavity_error',
                                'concave_points_error',
                                'symmetry_error',
                                'fractal_dimension_error',
                                'worst_smoothness',
                                'worst_symmetry',
                                'worst_fractal_dimension']
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("Num_pipeline",numerical_pipeline,numerical_columns)
                ]
            )
            logging.info('Data transformation completed successfully')
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def remove_outliers_IQR(self,col,df):
        '''
            Desc: This function will handle outliers in data with 5 number summary
        '''
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            iqr = Q3 - Q1
            
            higher_fence = Q3 + 1.5 * (iqr)
            lower_fence = Q1 - 1.5 * (iqr)
            
            df.loc[df[col]>higher_fence,col] = higher_fence
            df.loc[df[col]<lower_fence,col] = lower_fence

            return df
        
        except Exception as e:  
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data  = pd.read_csv(test_path) 
            numerical_columns = ['mean_radius',
                                'mean_texture',
                                'mean_smoothness',
                                'mean_compactness',
                                'mean_symmetry',
                                'mean_fractal_dimension',
                                'radius_error',
                                'texture_error',
                                'smoothness_error',
                                'compactness_error',
                                'concavity_error',
                                'concave_points_error',
                                'symmetry_error',
                                'fractal_dimension_error',
                                'worst_smoothness',
                                'worst_symmetry',
                                'worst_fractal_dimension']
            logging.info(f"Removing outlier from train data")
            for col in numerical_columns:
                self.remove_outliers_IQR(col=col,df=train_data)
            logging.info(f"Removing outlier from test data")
            for col in numerical_columns:
                self.remove_outliers_IQR(col=col,df=test_data)  
            
            preprocessor = self.get_data_transformation_obj()
            target_column = 'target'
            drop_column = [target_column]

            logging.info('Splitting train data into dependent and independent feature')
            input_feature_train_data = train_data.drop(drop_column,axis=1)
            target_feature_train_data = train_data[target_column]

            logging.info('Splitting test data into dependent and independent feature')
            input_feature_test_data = test_data.drop(drop_column,axis=1)
            target_feature_test_data = test_data[target_column]
            logging.info('Transforming data')
            input_train_arr = preprocessor.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor.transform(input_feature_test_data)
            logging.info('Concatenation transform data with target feature')
            train_arr = np.c_[input_train_arr,np.array(target_feature_train_data)]
            test_arr = np.c_[input_test_arr,np.array(target_feature_test_data)]
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor)

            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                )
        except Exception as e:
            raise CustomException(e,sys)
