import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle
df = pd.read_csv('BRCA.csv')
"""
#for EDA report
import pandas_profiling as pp
report = pp.ProfileReport(df)
report.to_file('Report.html')
"""
print("**Extraction of first 5 columns**")
print(df.head())
print('_____')
# null values
df_null_values = df.isnull().sum()
print('**Extraction of null values**')
print(df_null_values)
print('_____')
# drop null values
df = df.dropna()
print('**df info after removing null values**')
print(df.info())
print('_____')
# Summary
print('**Extraction of summary**')
print(df.describe(include="all"))
print('_____')
print('**Extraction of value counts**')
#Gender
print(df.Gender.value_counts())
# ER status
print(df["ER status"].value_counts())
# PR status
print(df["PR status"].value_counts())
# HER2 status
print(df["HER2 status"].value_counts())
print('_____')
## Exploratory df analysis
# Tumour Stage
stage = df["Tumour_Stage"].value_counts()
transactions = stage.index
quantity = stage.values

figure = px.pie(df,
             values=quantity,
             names=transactions,hole = 0.5,
             title="Tumour Stages of Patients")
#figure.show()
# Histology
histology = df["Histology"].value_counts()
transactions = histology.index
quantity = histology.values
figure = px.pie(df,
             values=quantity,
             names=transactions,hole = 0.5,
             title="Histology of Patients")
#figure.show()
# Surgery_type
surgery = df["Surgery_type"].value_counts()
transactions = surgery.index
quantity = surgery.values
figure = px.pie(df, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Type of Surgery of Patients")
#figure.show()
## df preparation
print("**Converting categorical variables to numeric**")
df["Tumour_Stage"] = df["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
df["Histology"] = df["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
df["ER status"] = df["ER status"].map({"Positive": 1})
df["PR status"] = df["PR status"].map({"Positive": 1})
df["HER2 status"] = df["HER2 status"].map({"Positive": 1, "Negative": 2})
df["Gender"] = df["Gender"].map({"MALE": 0, "FEMALE": 1})
df["Surgery_type"] = df["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})
df["Patient_Status"] = df["Patient_Status"].map({"Alive": 1, "Dead": 0})
print(df.head())
print('_____')
# Splitting data
y = df['Patient_Status']
x = df[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4',
                   'Tumour_Stage', 'Histology', 'ER status', 'PR status',
                   'HER2 status', 'Surgery_type']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
# Define SVM model and hyperparameters to tune
model = SVC()
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
# Use grid search to find best hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(x_train, y_train)
print('**Hyperparameter tuning**')
print("Best hyperparameters:", grid_search.best_params_)
# Train SVM model on full training set using best hyperparameters
model = SVC(**grid_search.best_params_)
print('_____')
model.fit(x_train, y_train)
# Evaluate performance on test set
y_pred = model.predict(x_test)
print("**Accuracy score**")
score = accuracy_score(y_test,y_pred)
print("Accuracy on test set:", int(score*100))
print('_____')
# Make predictions on new data
print("**Predictions on new data**")
new_features = [36.0, 1, 0.080353, 0.42638, 0.54715, 0.273680, 3, 1, 1, 1, 2, 2]
print('New Features : '+ str(new_features))
prediction = model.predict([new_features])
if prediction == 1:
    print("Prediction : Alive")
else:
    print("Prediction : Dead")
# Pickling of model
pickle.dump(model,open('model.pkl','wb'))