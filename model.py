import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# Select and preprocess features
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=35)

# Use GridSearchCV to tune hyperparameters
params = {
    'n_estimators': [100, 150],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 4]
}

grid = GridSearchCV(RandomForestClassifier(random_state=35), param_grid=params, cv=3)
grid.fit(X_train, y_train)

# Predict and evaluate
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Improved Accuracy:", accuracy)

# Save model and scaler
joblib.dump(grid.best_estimator_, "model.joblib")
joblib.dump(scaler, "scaler.joblib")








# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# import joblib
#
#
# # داده نمونه
# url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
# df = pd.read_csv(url)
#
# df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]
# df = df.dropna()
# df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
#
# X = df[["Pclass", "Sex", "Age", "Fare"]]
# y = df["Survived"]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
#
# model = RandomForestClassifier()
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
#
# joblib.dump(model, "model.joblib")