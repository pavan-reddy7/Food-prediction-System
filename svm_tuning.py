import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load V2 features
df = pd.read_csv(r"C:\ML_project\food_features_v2.csv")

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Same split used in all previous experiments
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Values to test
C_values = [0.1, 1, 10, 50, 100]

results = []

print("Starting SVM hyperparameter tuning...\n")

for C in C_values:

    print(f"Testing C = {C}")

    model = SVC(
        kernel="rbf",
        C=C
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    results.append((C, accuracy))

    print(f"Accuracy = {accuracy:.4f}\n")


# Display results
print("================================")
print("SVM HYPERPARAMETER RESULTS")
print("================================")

best_C = None
best_accuracy = 0

for C, accuracy in results:

    print(f"C = {C:<6} Accuracy = {accuracy:.4f}")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_C = C

print("\nBest C:", best_C)
print("Best Accuracy:", round(best_accuracy, 4))