import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data = train_data.drop('Employee ID', axis=1)
test_data = test_data.drop('Employee ID', axis=1)

le = LabelEncoder()

ordinal_cols = ['Gender', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 
                'Overtime', 'Education Level', 'Marital Status', 'Job Level', 
                'Company Size', 'Remote Work', 'Leadership Opportunities', 
                'Innovation Opportunities', 'Company Reputation', 'Employee Recognition', 'Attrition']

for col in ordinal_cols:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col]) if set(test_data[col]) <= set(le.classes_) else -1

train_data = pd.get_dummies(train_data, columns=['Job Role'], drop_first=True)
test_data = pd.get_dummies(test_data, columns=['Job Role'], drop_first=True)

test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

X_train = train_data.drop('Attrition', axis=1)
y_train = train_data['Attrition'].apply(lambda x: 1 if x == 'Left' else 0)

X_test = test_data.drop('Attrition', axis=1)
y_test = test_data['Attrition'].apply(lambda x: 1 if x == 'Left' else 0)

print(f'Training set class distribution:\n{y_train.value_counts()}')
print(f'Test set class distribution:\n{y_test.value_counts()}')

if y_train.value_counts().get(1, 0) == 0:
    print("No attrition (1) in the training set. Adjusting data preprocessing...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'lambda': [0, 1],
    'alpha': [0, 0.5]
}

scale_pos_weight = y_train.value_counts().get(0, 1) / y_train.value_counts().get(1, 1)
xgb_model = xgb.XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight)

stratified_kfold = StratifiedKFold(n_splits=5)

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=stratified_kfold, n_jobs=-1, verbose=2, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best parameters: ", grid_search.best_params_)

best_model = grid_search.best_estimator_

cv_scores = cross_val_score(best_model, X_train, y_train, cv=stratified_kfold)
print(f'Cross-validated accuracy on training data: {cv_scores.mean()}')

y_pred = best_model.predict(X_test)

print(f'Accuracy on test data: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Stayed', 'Left'], yticklabels=['Stayed', 'Left'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

print(f"Precision (Left): {precision_score(y_test, y_pred)}")
print(f"Recall (Left): {recall_score(y_test, y_pred)}")
print(f"F1-Score (Left): {f1_score(y_test, y_pred)}")

roc_auc = roc_auc_score(y_test, y_pred)
print(f'ROC AUC: {roc_auc}')

importances = best_model.feature_importances_
sorted_idx = importances.argsort()

plt.figure(figsize=(10, 8))
plt.barh(train_data.columns[sorted_idx], importances[sorted_idx], color='steelblue')
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.show()
