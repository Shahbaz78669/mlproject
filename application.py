from src.components.data_ingestion import DataIngestionConfig
from src.components.data_ingestion import DataIngestion
from src.utils import save_object
from  flask import Flask,render_template,request
import dill
import numpy as np
import pandas as pd
from src.pipeline.prediction_pipeline import CustomData
from src.pipeline.prediction_pipeline import predictPipeline

from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predictmarks',methods=['GET','POST'])
def predict_marks():
    if(request.method=='GET'):
       return render_template('home.html')
    else:
        data=CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = float(request.form.get('reading_score')),
            writing_score = float(request.form.get('writing_score'))
        )
        pred_data_df=data.get_input_to_data_frame()
        print(pred_data_df)


        predict_pipeline=predictPipeline()
        results=predict_pipeline.predict(pred_data_df)
        
        # Cap the score between 0 and 100
        predicted_score = min(100.0, max(0.0, results[0]))
        
        return render_template('home.html',results=predicted_score)



if __name__=='__main__':
 app.run(host='0.0.0.0')


















