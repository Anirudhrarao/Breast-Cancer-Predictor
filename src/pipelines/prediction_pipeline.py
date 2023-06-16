import os
import sys

import pandas as pd 
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from dataclasses import dataclass

class PredictionPipeline:
    def __init__(self):
        pass 

    def predict(self,features):
        '''
            Desc: This function will load pickle file and transform data and make prediction on give 
                  input features
        '''
        preprocessor_path = os.path.join('artifact/data_transformation','preprocessor.pkl')
        model_path = os.path.join('artifact/model_trainer','model.pkl')
        preprocessor = load_object(preprocessor_path)
        model = load_object(model_path)
        scaled = preprocessor.transform(features)
        pred = model.predict(scaled)
        return pred
    

class CustomClass:
    def __init__(self,mean_radius:float,
                mean_texture:float,
                mean_smoothness:float,
                mean_compactness:float,
                mean_symmetry:float,
                mean_fractal_dimension:float,
                radius_error:float,
                texture_error:float,
                smoothness_error:float,
                compactness_error:float,
                concavity_error:float,
                concave_points_error:float,
                symmetry_error:float,
                fractal_dimension_error:float,
                worst_smoothness:float,
                worst_symmetry:float,
                worst_fractal_dimension:float):
        self.mean_radius = mean_radius
        self.mean_texture = mean_texture
        self.mean_smoothness = mean_smoothness
        self.mean_compactness = mean_compactness
        self.mean_symmetry = mean_symmetry
        self.mean_fractal_dimension = mean_fractal_dimension
        self.radius_error = radius_error
        self.texture_error = texture_error
        self.smoothness_error = smoothness_error
        self.compactness_error = compactness_error
        self.concavity_error = concavity_error
        self.concave_points_error = concave_points_error
        self.symmetry_error = symmetry_error
        self.fractal_dimension_error = fractal_dimension_error
        self.worst_smoothness = worst_smoothness
        self.worst_symmetry = worst_symmetry
        self.worst_fractal_dimension = worst_fractal_dimension

    def get_data_as_dataFrame(self):
        try:
            custom_input = {
                'mean_radius':[self.mean_radius],
                'mean_texture':[self.mean_texture],
                'mean_smoothness':[self.mean_smoothness],
                'mean_compactness':[self.mean_compactness],
                'mean_symmetry':[self.mean_symmetry],
                'mean_fractal_dimension':[self.mean_fractal_dimension],
                'radius_error':[self.radius_error],
                'texture_error':[self.texture_error],
                'smoothness_error':[self.smoothness_error],
                'compactness_error':[self.compactness_error],
                'concavity_error':[self.concavity_error],
                'concave_points_error':[self.concave_points_error],
                'symmetry_error':[self.symmetry_error],
                'fractal_dimension_error':[self.fractal_dimension_error],
                'worst_smoothness':[self.worst_smoothness],
                'worst_symmetry':[self.worst_symmetry],
                'worst_fractal_dimension':[self.worst_fractal_dimension]
            }
            data = pd.DataFrame(custom_input)
            print(data)
            return data
        except Exception as e:
            raise CustomException(e,sys)
        
        
