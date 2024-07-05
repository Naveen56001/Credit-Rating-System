from flask import Flask, request, jsonify, render_template
from joblib import load
import pandas as pd
from flask_cors import CORS

# Load the trained model and LabelEncoders
model = load('D:\VIT\SEM 2\LAB\machine\PROJECT\Project_Model.pkl')

# Load the label encoders as a dictionary
label_encoders = load('D:\VIT\SEM 2\LAB\machine\PROJECT\Project_Label_Encoders.pkl')

app = Flask(__name__)

CORS(app)
# print(help(label_encoders))

@app.route('/')
def home():
    return 'Server running'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the request
    data = request.json

    # Convert data to DataFrame
    input_data = pd.DataFrame(data, index=[0])

    # feature =['CHK_ACCT', 'History', 'Purpose of credit', 'Balance in Savings A/C', 'Employment', 'Marital status', 'Co-applicant', 'Real Estate', 'Other installment', 'Residence', 'Job', 'Phone', 'Foreign', 'Credit classification']

    # Apply LabelEncoder transformation to categorical features
    for feature, encoder in label_encoders.items():
        if feature in input_data.columns:
            input_data[feature] = encoder.transform(input_data[feature])
    # input_data = label_encoders.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
