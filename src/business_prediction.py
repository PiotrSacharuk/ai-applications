import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

warnings.simplefilter("ignore")

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / 'data'

# data analytics
df = pd.read_excel(DATA_DIR / 'Coffee_Shop_data.xlsx')
population = pd.read_csv(DATA_DIR / 'population.csv', skiprows=[0])

print(f"{population.head()=}")
print(f"{df.head()=}")
print(f"{df.info()=}")

print(df.shape)
print(population.shape)

print(f"{df.describe()=}")

ax = df['City'].value_counts().head(5).plot(kind='bar')
ax.set_title('Top 5 Cities with mostCoffee Shops')
#plt.show()

ax= df['Business Name'].value_counts().head(10).plot(kind='bar')
ax.set_title('Top 10 most famous brands')
#plt.show()

# data preprocessing
print(df.isna().sum())

df['Zip Code'] = df['Zip Code'].astype(str)

def find_zip_code(geocode):
    pattern = r'\d{5}$'
    match = re.search(pattern, geocode)

    if match:
        zip_code = match.group(0)
    return zip_code

population['Zip Code'] = population['Geography'].apply(find_zip_code)

cafe_data = df.copy()
df = pd.merge(cafe_data, population)

columns = cafe_data.columns.values.tolist()+['Total']
df = df[columns]
df = df.rename(columns={'Total':'Population'})

print(df)
df = df[['Zip Code', 'Rating', 'Median Salary', 'Latte Price', 'Population']]

coffe_shop_counts = df['Zip Code'].value_counts().reset_index()
coffe_shop_counts.columns = ['Zip Code', 'Coffee Shop Count']

df['Zip Code'] = df['Zip Code'].astype(str)
coffe_shop_counts['Zip Code'] = coffe_shop_counts['Zip Code'].astype(str)

df = df.merge(coffe_shop_counts, on='Zip Code', how='left')

print(df)

sorted_df = df.sort_values(by=['Population', 'Coffee Shop Count', 'Rating', 'Median Salary'],
                           ascending=[False, True, True, False]).reset_index(drop=True)

print(sorted_df)

lst=[]
for i in range(len(sorted_df)):
    if len(lst) != 5:
        if (sorted_df['Zip Code'][i] not in lst):
            lst.append(sorted_df['Zip Code'][i])

top5_zip_codes_df = sorted_df[sorted_df['Zip Code'].isin(lst)]

print(top5_zip_codes_df)

X = df.drop(['Latte Price', 'Zip Code'], axis=1)
Y = df['Latte Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Model selection
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

# Hyper parameter tuning
param_grid = {
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'Gradient Boosting': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}, 'max_depth': [3, 5, 10],
}

for model_name, model in models.items():
    if model_name in param_grid:
        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, Y)
        models[model_name] = grid_search.best_estimator_


# Model training
for model_name, model in models.items():
    model.fit(X_train, Y_train)

# Model evaluation
for model_name, model in models.items():
    Y_pred = model.predict(X_test)

    print(f"{model_name} Metrics:")
    print(f"Mean Absolute Error: {mean_absolute_error(Y_test, Y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred)}")
    print(f"R^2 Score: {r2_score(Y_test, Y_pred)}")
    print("\n")


# Predictions
zip_codes_df = top5_zip_codes_df.drop(['Zip Code', 'Latte Price'], axis=1)
zip_codes_df = sc.transform(zip_codes_df)

for model_name, model in models.items():
    predicted_prices = model.predict(zip_codes_df)
    print(f"Predicted Latte Prices by {model_name} for Top 5 Zip Codes:")
    print(predicted_prices)
    print("\n")

predictions = {}
for model_name, model in models.items():
    predicted_prices = model.predict(zip_codes_df)
    predictions[model_name] = predicted_prices


predictions_df = pd.DataFrame(predictions)
predictions_df['Zip Code'] = top5_zip_codes_df['Zip Code'].values

cols = ['Zip Code'] + [col for col in predictions_df.columns if col != 'Zip Code']
predictions_df = predictions_df[cols]

agg_df = predictions_df.groupby('Zip Code')['Linear Regression'].agg([("Highest", "max"), ("Lowest", "min")]).reset_index()
agg_df.columns= ['Zip Code', 'Highest Price', 'Lowest Price']
print(agg_df)
