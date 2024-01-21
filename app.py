# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
from flask import Flask, request, render_template
import pickle
app = Flask("__name__")
df1=pd.read_csv('train.csv')
@app.route("/")
def loadpage():
    return render_template('home.html',query="")
@app.route("/",methods=['POST'])
def predict():
    '''
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    tenure,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod,
    MonthlyCharges,
    TotalCharges
    '''
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']
    inputQuery17 = request.form['query17']
    inputQuery18 = request.form['query18']
    inputQuery19 = request.form['query19']
    
    #loading random forest classifier
    model = pickle.load(open("rf_model.sav", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]
    
    new_df = pd.DataFrame(data, columns = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService","MultipleLines",                           
           "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
           "StreamingMovies","Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges"])
    
    df_2 = pd.concat([df1, new_df], ignore_index = True)
    #apply the preprocessing transformations to get the data ready for inference
    #encoding yes/no columns
    yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling']
    for col in yes_no_columns:
        df_2[col].replace({'Yes':1,'No':0},inplace=True)
    
    #encoding gender column
    df_2['gender'].replace({'Female':1,'Male':0},inplace=True)
    #One hot encoding for multi-categorical features
    #we will use pd.get_dummies()
    df_2=pd.get_dummies(data=df_2,columns=['InternetService','Contract','PaymentMethod'])
    
    #debugging
    print(len(df_2.columns))
    print(df_2.columns)
    
    #prediction
    single=model.predict(df_2.tail(1))
    proba=model.predict_proba(df_2.tail(1))[:,1]
    print("Prediction details:",single,"\nconfidence= ",model.predict_proba(df_2.tail(1)))
    
    if single==1:
        out1="This customer is likely to churn! "
        out2="Confidence: {}".format(proba*100)
    else:
        out1="This customer is likely to continue!"
        out2="Confidence: {}".format(100-proba*100)
    return render_template('home.html', output1=out1, output2=out2, 
                           query1 = request.form['query1'], 
                           query2 = request.form['query2'],
                           query3 = request.form['query3'],
                           query4 = request.form['query4'],
                           query5 = request.form['query5'], 
                           query6 = request.form['query6'], 
                           query7 = request.form['query7'], 
                           query8 = request.form['query8'], 
                           query9 = request.form['query9'], 
                           query10 = request.form['query10'], 
                           query11 = request.form['query11'], 
                           query12 = request.form['query12'], 
                           query13 = request.form['query13'], 
                           query14 = request.form['query14'], 
                           query15 = request.form['query15'], 
                           query16 = request.form['query16'], 
                           query17 = request.form['query17'],
                           query18 = request.form['query18'], 
                           query19 = request.form['query19'])

app.run()