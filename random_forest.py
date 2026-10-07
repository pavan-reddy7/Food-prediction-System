import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Load feature dataset
df = pd.read_csv(r"C:\ML_project\food_features.csv")

# Separate features and labels
X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Convert class names into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Same 80/20 stratified split as SVM
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

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

# Train
print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Predict
print("Making predictions...")
y_pred = model.predict(X_val)

# Accuracy
accuracy = accuracy_score(y_val, y_pred)

print("\nRandom Forest Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(
    classification_report(
        y_val,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)