from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bestatml import BestatClassifier

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = BestatClassifier(
    random_state=42,
    tune_hyperparameters=False,
    search_budget="lightweight",
    verbose=1,
)
model.fit(X_train, y_train)
print(model)
print(model.summary())
print("accuracy:", model.score(X_test, y_test))
print(model.predict_proba(X_test)[:3])
