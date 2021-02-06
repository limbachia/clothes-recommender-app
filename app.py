import numpy as np
from flask import Flask, request, jsonify, render_template

from recommender_model.recommender import Recommender

app = Flask(__name__)
recommender = Recommender()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    '''
    For rendering results on HTML GUI
    '''
    user_info = [x for x in request.form.values()]
    recommender.get_user_info(user_info)
    recommender.transform()
    
    output = recommender.get_recommendations()
    
    return render_template('index.html', 
                           tables=[output.to_html(classes='data')])

if __name__ == "__main__":
    app.run(debug=True)