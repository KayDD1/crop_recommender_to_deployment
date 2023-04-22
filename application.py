import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,PowerTransformer
from src.pipeline.predict_pipeline import CustomData,PredictionPipeline

application=Flask(__name__)
app=application

# Create route for homepage

@app.route('/')
def index():
     return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])

def predict():
    if request.method=='GET':
         return render_template('home.html')
    else:
     data=CustomData(
          ## get the input values from webpage
          K = float(request.form.get('K')),
          N = float(request.form.get('N')),
          P = float(request.form.get('P')),
          temperature = float(request.form.get('temperature')),
          rainfall = float(request.form.get('rainfall')),
          humidity = float(request.form.get('humidity')),
          ph = float(request.form.get('ph'))
     )
     pred_df=data.get_input_as_dataframe()
     print(pred_df)

     predict_pipeline=PredictionPipeline()
     results=predict_pipeline.predict(pred_df)
     print(results[0])
     return render_template('home.html',results=results[0])


if __name__=="__main__":
     app.run(host="0.0.0.0")




        
