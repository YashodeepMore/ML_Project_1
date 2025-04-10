import pickle
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline, CustomData


application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoints():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity = request.form.get('ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )
        df = data.get_data_as_data_frame()
        print(df)

        predict_pipeline = PredictPipeline()
        result=predict_pipeline.predict(df)
        return render_template('home.html',results=result[0])


if __name__ == "__main__":
    app.run(debug=True)
