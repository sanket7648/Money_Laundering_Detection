import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
data = pd.read_csv(r'E:\newww\creditcard.csv')

# Split into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Dictionary to store model scores
model_scores = {}

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
pickle.dump(log_model, open('log_model.pkl', 'wb'))
y_pred = log_model.predict(X_test)
model_scores['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])
}

# Linear Regression (not recommended for classification, but included as per original code)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
pickle.dump(lin_model, open('lin_model.pkl', 'wb'))
y_pred = lin_model.predict(X_test).round()
model_scores['Linear Regression'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, y_pred)
}

# KNN
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
pickle.dump(knn_model, open('knn_model.pkl', 'wb'))
y_pred = knn_model.predict(X_test)
model_scores['KNN'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, knn_model.predict_proba(X_test)[:, 1])
}

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
pickle.dump(nb_model, open('nb_model.pkl', 'wb'))
y_pred = nb_model.predict(X_test)
model_scores['Naive Bayes'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, nb_model.predict_proba(X_test)[:, 1])
}

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
y_pred = rf_model.predict(X_test)
model_scores['Random Forest'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
}

# SVM
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)
pickle.dump(svm_model, open('svm_model.pkl', 'wb'))
y_pred = svm_model.predict(X_test)
model_scores['SVM'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
}

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)
pickle.dump(xgb_model, open('xgb_model.pkl', 'wb'))
y_pred = xgb_model.predict(X_test)
model_scores['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
}

# Deep Neural Network (DNN)
dnn_model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)
dnn_model.save('dnn_model.h5')

y_pred = (dnn_model.predict(X_test) > 0.5).astype('int32')
model_scores['DNN'] = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1 Score': f1_score(y_test, y_pred),
    'AUC Score': roc_auc_score(y_test, dnn_model.predict(X_test))
}

# Print performance metrics for all models
print("\nPerformance Metrics for All Models:")
for model_name, scores in model_scores.items():
    print(f"\n{model_name}:")
    for metric, value in scores.items():
        print(f"  {metric}: {value:.4f}")
