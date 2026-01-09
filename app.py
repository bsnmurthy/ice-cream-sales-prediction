from flask import Flask, request, render_template
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Training model
X = np.array([[20],[22],[25],[27],[30],[32],[35],[37],[40]])
y = np.array([100,120,150,170,200,230,270,300,340])

model = LinearRegression()
model.fit(X, y)

@app.route('/')
def home():
    return '''
    <h2>Ice Cream Sales Prediction</h2>
    <form action="/predict" method="post">
        Temperature: <input type="number" name="temp">
        <input type="submit">
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    temp = float(request.form['temp'])
    prediction = model.predict([[temp]])
    return f"<h3>Predicted Sales: {int(prediction[0])}</h3>"

if __name__ == '__main__':
    app.run(debug=True)
