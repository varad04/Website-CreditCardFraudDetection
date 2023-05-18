from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    file = request.files['csv_file']
    if not file:
        return "No file uploaded"
    
    df = pd.read_csv(file)
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Training logistic regression model
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    # Training random forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    return render_template('result.html', lr_accuracy=lr_accuracy, rf_accuracy=rf_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
