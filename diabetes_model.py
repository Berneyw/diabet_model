import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

diabetes_data_path = "data/diabetes.csv"
diabetes_data = pd.read_csv(diabetes_data_path)

diabetes_data = diabetes_data.dropna(axis=0)

result = diabetes_data.Outcome

train_result, val_result, train_data, val_data = train_test_split(
    result, diabetes_data, train_size=0.8, test_size=0.2, random_state=0
)

# model = RandomForestRegressor(random_state=0)
model = RandomForestClassifier(random_state=0)
model.fit(train_data, train_result)

print("Making prediction...")
prediction = model.predict_proba(val_data)[:, 1]
print(val_data.head())
print(f"Predictions is: {prediction}")

accuracy = accuracy_score(val_result, model.predict(val_data))
print(f"Accuracy: {accuracy * 100:.2f}%")

