from flask import Flask, request, render_template
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipline.predict_pipeline import CustomData, PredictPipeline
sys.path.append("/Users/reetu/Documents/Projects/AutopricingnewC2B/src/components/")
application = Flask(__name__)

app = application

## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template('home.html')
    else:
        data = CustomData(
            brand=request.form.get('brand'),
            model=request.form.get('model'),
            model_year=int(request.form.get('model_year')),
            milage=request.form.get('milage'),
            fuel_type=request.form.get('fuel_type'),
            engine=request.form.get('engine'),
            transmission=request.form.get('transmission'),
            ext_col=request.form.get('ext_col'),
            int_col=request.form.get('int_col'),
            accident=request.form.get('accident'),
            clean_title=request.form.get('clean_title'),
            price=request.form.get('price')
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        log_results = predict_pipeline.predict(pred_df)

        # Convert log-transformed price back to original price
        original_price = np.exp(log_results[0])
        
        return render_template('home.html', results=original_price)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
