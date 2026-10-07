import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------------
# 1. Load dataset
# -----------------------------------------

data_path = r"C:\ML_project\food_features.csv"

df = pd.read_csv(data_path)

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]


# -----------------------------------------
# 2. Train / Validation split
# -----------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 3. Feature scaling
# -----------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# -----------------------------------------
# 4. Train SVM
# -----------------------------------------

print("Training SVM...")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))

svm_model = SVC(
    kernel="rbf",
    C=10
)

svm_model.fit(X_train_scaled, y_train)


# -----------------------------------------
# 5. Make predictions
# -----------------------------------------

print("\nMaking predictions...")

y_pred = svm_model.predict(X_val_scaled)


# -----------------------------------------
# 6. Evaluate
# -----------------------------------------

accuracy = accuracy_score(y_val, y_pred)

print("\n----------------------------")
print("SVM RESULTS")
print("----------------------------")

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)

print("\nClassification Report:")
print(classification_report(y_val, y_pred))