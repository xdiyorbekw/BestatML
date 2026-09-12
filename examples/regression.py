from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from bestatml import BestatRegressor

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = BestatRegressor(
    random_state=42,
    tune_hyperparameters=False,
    ensemble_method="weighted_blending",
)
model.fit(X_train, y_train)
print(model)
print("r2:", model.score(X_test, y_test))
