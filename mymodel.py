import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib


df = pd.read_csv('german.data-numeric', sep='\s+', header=None)
df.columns = [f'Feature_{i}' for i in range(1, 26)]
df['Credit_risk'] = df['Feature_25'].map({1: 1, 2: 0}) 
df.drop('Feature_25', axis=1, inplace=True)


X = df.drop('Credit_risk', axis=1)
y = df['Credit_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'credit_model.pkl')

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

def predict(input_df):
    model = joblib.load('credit_model.pkl')
    return model.predict(input_df), model

    