import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load V2 CNN features
df = pd.read_csv(r"C:\ML_project\cnn_features_v2.csv")

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode food classes
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Same 80/20 stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Standardize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Logistic Regression
model = LogisticRegression(
    C=10,
    max_iter=2000
)

print("Training Logistic Regression on V2 CNN features...")

model.fit(X_train, y_train)

y_pred = model.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("\n================================")
print("CNN V2 + LOGISTIC REGRESSION")
print("================================")
print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)