from src.exception import CustomException
from src.logger import logging
import sys
import os
from src.utils import save_object
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import DataIngestionConfig
from src.utils import load_object



class predictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path='artifacts\model_trainer.pkl'
            preprocessor_path='artifacts\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
             raise CustomException(e,sys)
        




class CustomData:
    def __init__(self,gender:str,race_ethnicity:str,parental_level_of_education:str ,lunch:str,test_preparation_course:str,reading_score:int,writing_score:int):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score

    def get_input_to_data_frame(self):
        try:
                data_input_as_dict={
                    "gender":[self.gender],
                    "race_ethnicity":[self.race_ethnicity],
                    "parental_level_of_education":[self.parental_level_of_education],
                    "lunch":[self.lunch],
                    "test_preparation_course":[self.test_preparation_course],
                    "reading_score":[self.reading_score],
                    "writing_score":[self.writing_score],



                }
                return pd.DataFrame(data_input_as_dict)
        except Exception as e:
             raise CustomException(e,sys)

        