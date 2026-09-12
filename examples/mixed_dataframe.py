import pandas as pd
from sklearn.model_selection import train_test_split

from bestatml import BestatClassifier

df = pd.DataFrame({
    "age": [21, 33, 42, None, 29, 51, 47, 26, 39, 58],
    "income": [25000, 61000, None, 72000, 48000, 99000, 81000, 33000, 55000, 103000],
    "city": ["Tashkent", "Samarkand", "Tashkent", "Bukhara", None, "Tashkent", "Bukhara", "Samarkand", "Tashkent", "Bukhara"],
    "member": [True, False, True, True, False, True, False, True, False, True],
    "created_at": pd.to_datetime(["2026-01-01", "2026-02-14", "2026-03-02", "2026-03-29", "2026-04-10", "2026-05-03", "2026-05-20", "2026-06-01", "2026-06-15", "2026-07-01"]),
})
y = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1]
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)

model = BestatClassifier(random_state=42, tune_hyperparameters=False, max_models=3)
model.fit(X_train, y_train)
print(model.selected_models_)
print(model.score(X_test, y_test))
