from flask import Flask, render_template, request
import pickle
import numpy as np

model1 = pickle.load(open('model/brainstroke.pkl', 'rb'))  

app = Flask(__name__)

@app.route("/", methods=['GET'])
def hello():
    return render_template('front.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST': 
        d1 = request.form['Gender']        
        d2 = request.form['Age']      
        d3 = request.form['hypertension']  # Convert to int if needed
        d4 = request.form['heart_disease'] 
        d5 = request.form['ever_married']
        d6 = request.form['avg_glucose_level'] # Convert to float if needed
        d7 = request.form['bmi']  # Convert to float if needed
        d8 = request.form['smoking_status'] # Convert to int if needed

         # Encoding dictionaries
        gender_dict = {'Male': 1, 'Female': 0}
        yes_no_dict = {'Yes': 1, 'No': 0}
        smoking_status={'Never':0,'Former':1,'Current':2,'Unknown':3}

        d1 = gender_dict[d1]
        d3 = yes_no_dict[d3]
        d4 = yes_no_dict[d4]
        d5 = yes_no_dict[d5]
        d8 = smoking_status[d8]        

        # Prepare the array for prediction
        arr = np.array([[d1, d2, d3, d4, d5, d6, d7, d8]])
        print([d1, d2, d3, d4, d5, d6, d7, d8])
        pred1 = model1.predict(arr)

        if pred1[0] == 0:
            pred1 = "Negative" 
        else:
            pred1 = "Positive"

    return render_template('output.html', prediction_text1=pred1)

if __name__ == '__main__':
    app.run(debug=True)
