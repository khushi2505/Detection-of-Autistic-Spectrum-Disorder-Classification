from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = 'ASD Detected' if prediction[0] == 1 else 'ASD Not Detected'
    
    return render_template('result.html', prediction=output)

if __name__ == "__main__":
    app.run(debug=True)
