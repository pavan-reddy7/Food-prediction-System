import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Load V2 features
df = pd.read_csv(r"C:\ML_project\food_features_v2.csv")

X = df.drop(columns=["food_class", "image_name"])
y = df["food_class"]

# Encode class labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

print("Number of samples:", len(X))
print("Number of features:", X.shape[1])
print("Number of classes:", len(encoder.classes_))

# Pipeline:
# Scaling is performed separately inside every fold
model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="linear",
        C=10
    ))
])

# 5-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nRunning 5-fold cross-validation...")

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

print("\n================================")
print("5-FOLD CROSS-VALIDATION RESULTS")
print("================================")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nMean Accuracy:", round(scores.mean(), 4))
print("Standard Deviation:", round(scores.std(), 4))