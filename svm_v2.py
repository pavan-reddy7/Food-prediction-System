import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load V2 features
df = pd.read_csv(r"C:\ML_project\food_features_v2.csv")

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode food class names
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# SAME split as previous experiments
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Number of features:", X_train.shape[1])
print("Number of classes:", len(label_encoder.classes_))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Same SVM settings as our baseline
model = SVC(
    kernel="rbf",
    C=10
)

print("\nTraining SVM with V2 features...")
model.fit(X_train_scaled, y_train)

print("Making predictions...")
y_pred = model.predict(X_val_scaled)

accuracy = accuracy_score(y_val, y_pred)

print("\nSVM V2 Accuracy:", round(accuracy, 4))

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)