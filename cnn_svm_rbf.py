import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load CNN features
df = pd.read_csv(r"C:\ML_project\cnn_features.csv")

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode classes
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Same 80/20 split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)


# Test RBF with different C values
C_values = [1, 10, 50]

results = []

print("CNN + RBF SVM tuning\n")

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


print("================================")
print("CNN + RBF SVM RESULTS")
print("================================")

for C, accuracy in results:
    print(f"C = {C:<4} Accuracy = {accuracy:.4f}")

best_C, best_accuracy = max(results, key=lambda x: x[1])

print("\nBest C:", best_C)
print("Best Accuracy:", round(best_accuracy, 4))
print("Best Accuracy (%):", round(best_accuracy * 100, 2), "%")