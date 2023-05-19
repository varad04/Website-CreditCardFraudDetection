from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    lr_precision = precision_score(y_test, lr_pred)
    lr_recall = recall_score(y_test, lr_pred)
    lr_f1_score = f1_score(y_test, lr_pred)
    
    # Training random forest model
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_precision = precision_score(y_test, rf_pred)
    rf_recall = recall_score(y_test, rf_pred)
    rf_f1_score = f1_score(y_test, rf_pred)
    
    # Training support vector machine model
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred)
    svm_precision = precision_score(y_test, svm_pred)
    svm_recall = recall_score(y_test, svm_pred)
    svm_f1_score = f1_score(y_test, svm_pred)
    
    # Training decision tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    dt_precision = precision_score(y_test, dt_pred)
    dt_recall = recall_score(y_test, dt_pred)
    dt_f1_score = f1_score(y_test, dt_pred)
    
    # Training k-nearest neighbors model
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    knn_pred = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_pred)
    knn_precision = precision_score(y_test, knn_pred)
    knn_recall = recall_score(y_test, knn_pred)
    knn_f1_score = f1_score(y_test, knn_pred)
    
    return render_template('result.html',
                           lr_accuracy=lr_accuracy,
                           lr_precision=lr_precision,
                           lr_recall=lr_recall,
                           lr_f1_score=lr_f1_score,
                           rf_accuracy=rf_accuracy,
                           rf_precision=rf_precision,
                           rf_recall=rf_recall,
                           rf_f1_score=rf_f1_score,
                           svm_accuracy=svm_accuracy,
                           svm_precision=svm_precision,
                           svm_recall=svm_recall,
                           svm_f1_score=svm_f1_score,
                           dt_accuracy=dt_accuracy,
                           dt_precision=dt_precision,
                           dt_recall=dt_recall,
                           dt_f1_score=dt_f1_score,
                           knn_accuracy=knn_accuracy,
                           knn_precision=knn_precision,
                           knn_recall=knn_recall,
                           knn_f1_score=knn_f1_score)

if __name__ == '__main__':
    app.run(debug=True)
