import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, RocCurveDisplay


# Read text input
file_name = "sba_small.csv"
db = pd.read_csv(file_name, delimiter=',', header=0)

print(db.head())

db_train = db[db.Selected > 0]
db_test = db[db.Selected == 0]

db_train.to_csv('train.csv', index=False)
db_test.to_csv('test.csv', index=False)

x_col = ['Recession', 'RealEstate', 'Portion']
y_col = ['Default']

# build logistic model
model = LogisticRegression()
model.fit(db_train[x_col], db_train[y_col])

# save model
joblib.dump(model, 'model.joblib')

# create confusion matrix
y_true = db_test[y_col]
y_pred = model.predict(db_test[x_col])
cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

# save confusion matrix
cm_str = f"'tn': {cm[0,0]}, 'fp': {cm[0,1]}, 'fn': {cm[1,0]}, 'tp': {cm[1,1]}"
cm_str = '{' + cm_str + '}'
with open('confusion_matrix.json', 'w') as f:
    json.dump(cm_str, f)

# get ROC
y_score = model.decision_function(db_test[x_col])
roc = roc_curve(y_true, y_score)

# plot ROC
fig = plt.figure()
# for idx in range(len(roc)):
RocCurveDisplay.from_predictions(
    y_true,
    y_score,
    name="ROC curve",
    color="darkorange"
)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
