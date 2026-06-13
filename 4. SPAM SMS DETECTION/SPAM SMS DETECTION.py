# ==========================================
# SPAM SMS DETECTION
# CodSoft Internship Task 4
# ==========================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

print("Loading Dataset...")

# ==========================================
# FIND DATASET
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_file = None

for file in os.listdir(BASE_DIR):
    if file.endswith(".csv"):
        csv_file = os.path.join(BASE_DIR, file)
        break

if csv_file is None:
    raise FileNotFoundError("No CSV file found!")

print("Dataset Found:", os.path.basename(csv_file))

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv(csv_file, encoding="latin-1")

print("\nDataset Loaded Successfully!")
print("Shape:", data.shape)

# ==========================================
# CLEAN DATASET
# ==========================================

data = data.iloc[:, :2]
data.columns = ["label", "message"]

print("\nColumns:")
print(data.columns)

# ==========================================
# CONVERT LABELS
# spam = 1
# ham = 0
# ==========================================

data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

# ==========================================
# FEATURES & TARGET
# ==========================================

X = data["message"]
y = data["label"]

# ==========================================
# TF-IDF VECTORIZATION
# ==========================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = tfidf.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ==========================================
# MODEL
# ==========================================

print("\nTraining Naive Bayes Model...")

model = MultinomialNB()

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# RESULTS
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("SPAM SMS DETECTION")
print("==============================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# CUSTOM SMS TEST
# ==========================================

while True:

    sms = input("\nEnter SMS Message: ")

    sms_vector = tfidf.transform([sms])

    prediction = model.predict(sms_vector)

    if prediction[0] == 1:
        print("Prediction: SPAM SMS")
    else:
        print("Prediction: LEGITIMATE SMS")

    choice = input("\nCheck Another SMS? (y/n): ")

    if choice.lower() != "y":
        break

print("\nProgram Finished Successfully!")