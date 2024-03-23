import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the saved gradient_boost_model model
try:
    with open('/workspaces/house-price-deployment/model/gradient_boost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print("Error: The model file 'gradient_boost_model.pkl' is not found.")
    exit(1)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for handling prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch user input values from the form
    property_type = request.form['property_type']
    baths = int(request.form['baths'])
    bedrooms = int(request.form['bedrooms'])
    total_area = float(request.form['total_area'])
    
    # Use the model to make predictions
    prediction_input = np.array([[baths, bedrooms, total_area]])
    predicted_price = model.predict(prediction_input)[0]
    
    return render_template('index.html', prediction_result=f'Predicted Price: ${predicted_price:.2f}')

if __name__ == '__main__':
   app.run(host="0.0.0.0", port=9090, debug=True)
