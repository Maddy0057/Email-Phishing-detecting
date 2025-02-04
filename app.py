from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

app = Flask(__name__, static_folder='.')

# Load models and vectorizer
nb_model = joblib.load('phishing_email_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Serve the HTML file directly
@app.route('/')
def home():
    return send_from_directory(os.getcwd(), 'index.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get email text from JSON body
        email_text = request.json.get('email')

        if not email_text:
            return jsonify({'error': 'Email text is missing'}), 400

        # Vectorize and predict
        email_vec = vectorizer.transform([email_text])
        prediction = nb_model.predict(email_vec)

        result = {'prediction': 'Phishing' if prediction[0] == 1 else 'Not Phishing'}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': 'An error occurred during prediction', 'details': str(e)}), 500

# Required for Vercel serverless function
def handler(request, context):
    from werkzeug.wrappers import Request
    req = Request(request.environ)
    return app(req.environ, context)
