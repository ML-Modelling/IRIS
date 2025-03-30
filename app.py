from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "iris_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# Initialize Flask app
app = Flask(__name__)

# Route for rendering the UI


@app.route('/')
def home():
    return render_template('index.html')

# Route for handling predictions


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values from the HTML
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Prepare input for model
        features = np.array(
            [sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        prediction = model.predict(features)

        # Iris flower categories
        species = ["Setosa", "Versicolor", "Virginica"]
        result = species[prediction[0]]

        return render_template('index.html', prediction_text=f'Predicted Flower: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


# Run the app
if __name__ == '__main__':
    app.run(debug=True)
