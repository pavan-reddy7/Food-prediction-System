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

# Same 80/20 split
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

# Test different kernels
kernels = ["linear", "poly", "rbf"]

results = []

print("Starting SVM kernel comparison...\n")

for kernel in kernels:

    print(f"Testing kernel = {kernel}")

    model = SVC(
        kernel=kernel,
        C=10
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)

    results.append((kernel, accuracy))

    print(f"Accuracy = {accuracy:.4f}\n")


print("================================")
print("SVM KERNEL COMPARISON")
print("================================")

for kernel, accuracy in results:
    print(f"{kernel:<8} Accuracy = {accuracy:.4f}")