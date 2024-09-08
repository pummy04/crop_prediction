from flask import Flask , render_template,url_for,request
import pandas as pd 
import joblib 
import sqlite3
from pymongo import MongoClient

file_path = r'C:\Users\dell\Desktop\farmer\models\standardscaler.lb'
std_scaler = joblib.load(file_path)


kmeans_model = joblib.load('./models/kmeans_model.lb')
df = pd.read_csv("./models/filter_crops.csv")
app = Flask(__name__) 
 
@app.route('/')
def home():
    return render_template('home.html') 

@app.route('/output.html')
def output():
    return render_template('output.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/thank_you.html')
def thank_you():
    return render_template('thank_you.html')



@app.route('/predict',methods=['GET','POST'])
def predict(): 
    if request.method == 'POST': 
        N = int(request.form['N'])
        PH = float(request.form['PH'])
        P = int(request.form['P'])
        K = int(request.form['K'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])   # type == number >> text 
        UNSEEN_DATA = [[N,P,K,temperature,humidity,PH,rainfall]] 
        transformed_data = std_scaler.transform(UNSEEN_DATA)
        cluster = kmeans_model.predict(transformed_data)[0]
        suggestion_crops = list(df[df['cluster_no'] == cluster]['label'].unique())
        data = {"N":N,"P":P,"K":K,"temperature":temperature,"humidity":humidity,"PH":PH,"rainfall":rainfall}
       
        return render_template('output.html', crops=suggestion_crops)
    
   





if __name__ =="__main__":
    app.run(debug=True) 


  


  