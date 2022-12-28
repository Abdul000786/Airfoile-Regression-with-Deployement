import pickle
import flask 
from flask import Flask, request, app,jsonify,url_for,render_template
import numpy as np
import pandas as pd




app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route('/home',methods=['POST'])
def home():
    #return 'Hello world'
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    ##here we are not creating any html page  from the post man we post a specific request when this post will send and how do we capture this information that is why we have used this library request it helps you to capture the actual json data which is coming from the postman
    data=request.json['data'] 
    print(data) 
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)
    #this is only for postman means for testing our how our model is working if we want to hit this is api from anywhere or website then we will have to deployment on heroku , aws, azure, for this first we have to create two folders one is static and another is template , static and template are use for css style html and in template html file will go into this folder html file copied from link of codeshare is code share io /6pLPAk



@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_features=[np.array(data)] 
    print(data) 
    
    output=model.predict(final_features)[0]
    print(output)
    return render_template('home.html',prediction_text="Airfoil pressure is {}".format(output))


if __name__=="__main__":  #that is where  your point of execution will start of the entire code
    app.run(debug=True)

