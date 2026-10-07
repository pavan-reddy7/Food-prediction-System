import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Load feature dataset
data_path = r"C:\ML_project\food_features.csv"

df = pd.read_csv(data_path)

print("Original dataset shape:", df.shape)


# Separate features (X) and target (y)
X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

print("X shape:", X.shape)
print("y shape:", y.shape)


# Stratified 80/20 train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


print("\nScaled training shape:", X_train_scaled.shape)
print("Scaled validation shape:", X_val_scaled.shape)

print("\n----------------------------")
print("Data preparation complete!")
print("----------------------------")