import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# Load CNN features
df = pd.read_csv(r"C:\ML_project\cnn_features.csv")

# Separate features and labels
X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode food classes
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Training dataset:")
print("Samples:", len(X))
print("CNN features:", X.shape[1])
print("Classes:", len(encoder.classes_))


# Same 80/20 stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# Standardization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# Linear SVM
model = SVC(
    kernel="linear",
    C=10
)

print("\nTraining Linear SVM with CNN features...")

model.fit(X_train_scaled, y_train)

print("Making predictions...")

y_pred = model.predict(X_val_scaled)


# Accuracy
accuracy = accuracy_score(y_val, y_pred)

print("\n================================")
print("CNN + LINEAR SVM RESULTS")
print("================================")

print("Accuracy:", round(accuracy, 4))
print("Accuracy (%):", round(accuracy * 100, 2), "%")


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_val,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)