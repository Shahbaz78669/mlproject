import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import sys
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.utils import save_object
import dill




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_obj(self):

        try:
            numerical_column=[
                'reading_score',
                'writing_score'
            ]

            categorical_column=[
            'gender',
            'race_ethnicity',
            'parental_level_of_education',
            'lunch',

            'test_preparation_course'
            ]


            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='mean')),
                ("Scaler",StandardScaler())
                ]
            )

            logging.info("Numerical pipeline completed ")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehot",OneHotEncoder(sparse_output=False)),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info("Categorical pipeline completed")


            preprocessor=ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_column),
                    ('cat_pipeline',cat_pipeline,categorical_column)
                ]

            )

            logging.info("preprocessor object created ")


            return preprocessor




        except Exception as e:
            raise CustomException(e,sys)






    def initiate_data_transformation(self,train_file_path,test_file_path):
        try:
            preprocessor_obj=self.get_data_transformation_obj()
            logging.info("Obtained preprocessor object")

            train_df=pd.read_csv(train_file_path)
            logging.info("Read train data")
            test_df=pd.read_csv(test_file_path)
            logging.info("Read test data")

            numerical_column=[
                            'reading_score',
                            'writing_score'
                        ]
            
            categorical_column=[
                        'gender',
                        'race_ethnicity',
                        'parental_level_of_education',
                        'lunch',
            
                        'test_preparation_course'
                        ]

            train_df_input=train_df.drop('math_score',axis=1)
            train_df_target=train_df['math_score']

            test_df_input=test_df.drop('math_score',axis=1)
            test_df_target=test_df['math_score']


            logging.info("Applying preprocessor object to input train adn test")


            train_input_feature_arr=preprocessor_obj.fit_transform(train_df_input)
            test_input_feature_arr=preprocessor_obj.transform(test_df_input)


            train_arr=np.c_[
                 train_input_feature_arr, np.array(train_df_target)
            ]

            test_arr=np.c_[
                test_input_feature_arr,np.array(test_df_target)
            ]


            logging.info("saving preprocessing object")



            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )


            return( train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)




            











        except Exception as e:
            raise CustomException(e,sys)
            
        
            
        

        






    






    

