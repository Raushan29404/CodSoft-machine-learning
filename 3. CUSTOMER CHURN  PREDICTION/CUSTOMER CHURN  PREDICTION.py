import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading Dataset...")

# Dataset Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(BASE_DIR, "Churn_Modelling.csv")

# Load Dataset
data = pd.read_csv(csv_file)

print("Dataset Loaded Successfully!")
print("Shape:", data.shape)

# Remove unnecessary columns
drop_cols = ["RowNumber", "CustomerId", "Surname"]

for col in drop_cols:
    if col in data.columns:
        data.drop(columns=col, inplace=True)

# Encode all object/string columns
for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

# Check datatypes
print("\nData Types:")
print(data.dtypes)

# Target column
target = "Exited"

# Features and Target
X = data.drop(columns=[target])
y = data[target]

# Force numeric conversion
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values
X = X.fillna(0)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# Model
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("CUSTOMER CHURN PREDICTION")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nProgram Finished Successfully!")