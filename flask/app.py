from flask import Flask, request, render_template
import pickle
import numpy as np
import joblib

app = Flask(__name__)

# Load model and feature columns
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

feature_columns = joblib.load("feature_columns.pkl")

@app.route('/')
def home():
    return render_template('index.html', feature_columns=feature_columns)

@app.route('/predict', methods=['POST'])
def predict():
    # Get values in the same order as feature_columns
    input_data = [float(request.form.get(col)) for col in feature_columns]
    final_input = np.array([input_data])
    prediction = model.predict(final_input)
    
    result = "✅ Patient is likely to have Liver Cirrhosis" if prediction[0] == 1 else "❎ Patient is not likely to have Liver Cirrhosis"
    return render_template('index.html', prediction_text=result, feature_columns=feature_columns)

if __name__ == "__main__":
    app.run(debug=True, port=5002)
