import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
data = {
    'Name': ['Arun','Bala','Charan','Divya','Eswar','Fathima','Gokul','Hari','Indhu','Jagan'],
    'Age': [25,np.nan,30,28,np.nan,35,40,29,np.nan,32],
    'Salary': [50000,60000,np.nan,45000,70000,np.nan,80000,52000,61000,np.nan],
    'Experience': [2,5,np.nan,3,7,10,np.nan,4,6,8],
    'Department': ['IT','HR',np.nan,'Finance','IT','HR','IT','Finance',np.nan,'HR'],
    'City': ['Chennai','Madurai','Trichy',np.nan,'Chennai','Salem','Madurai','Trichy','Chennai',np.nan],
    'Gender': ['M','F','M','F','M',np.nan,'M','F','F','M'],
    'Performance_Score': [85,90,np.nan,88,92,95,89,np.nan,91,87]
}
df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)
df = df.drop_duplicates()
numeric_cols = ['Age','Salary','Experience','Performance_Score']
categorical_cols = ['Department','City','Gender']
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_cols),
    ('cat', categorical_pipeline, categorical_cols)
])
processed_data = preprocessor.fit_transform(df)

columns = (
    numeric_cols +
    list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols))
)
df_processed = pd.DataFrame(processed_data, columns=columns)
pd.set_option('display.max_columns', None)
print("\n Preprocessed Dataset:\n")
print(df_processed)