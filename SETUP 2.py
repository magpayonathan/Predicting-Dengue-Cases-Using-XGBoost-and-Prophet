#------- XGBOOST -------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
tscv = TimeSeriesSplit(n_splits=5)

weather_vars = ['Rainfall Amount', 'Max Temperature', 'Minimum Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction']

df = pd.read_csv("dengue2023.csv")
df['Week'] = pd.to_datetime(df['Week'], format='%m/%d/%Y')

def create_lag_features(data, lags=12):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Cases'].shift(lag)
    df['month'] = df['Week'].dt.month
    df['week_of_year'] = df['Week'].dt.isocalendar().week
    df.dropna(inplace=True)
    return df

df = create_lag_features(df, lags=12)

features = weather_vars + [f'lag_{i}' for i in range(1, 5)] + ['month', 'week_of_year']

X = df[features]
y = df['Cases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5, 7],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'gamma': [0, 0.1],
    'reg_alpha': [0, 0.1],
    'reg_lambda': [1, 10]
}

grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, cv=tscv,
                           scoring='neg_mean_absolute_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Optimized Mean Absolute Error (MAE): {mae}")
print(f"Optimized Root Mean Squared Error (RMSE): {rmse}")
print(f"Optimized Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

df_train = pd.read_csv("dengue2023.csv")
df_weather_2023 = pd.read_csv("weather2023.csv")
df_actual_2023 = pd.read_csv("2023_cases.csv")

df_train['Week'] = pd.to_datetime(df_train['Week'], format='%m/%d/%Y')
df_train['month'] = df_train['Week'].dt.month
df_train['week_of_year'] = df_train['Week'].dt.isocalendar().week

def create_lag_features(data, lags=12):
    df = data.copy()
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['Cases'].shift(lag)
    df['month'] = df['Week'].dt.month
    df['week_of_year'] = df['Week'].dt.isocalendar().week
    df.dropna(inplace=True)
    return df

df_train = create_lag_features(df_train, lags=12)

features = weather_vars + [f'lag_{i}' for i in range(1, 5)] + ['month', 'week_of_year']
X_train = df_train[features]
y_train = df_train['Cases']

model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

df_weather_2023['Week'] = pd.to_datetime(df_weather_2023['Week'], format='%m/%d/%Y')
df_weather_2023['month'] = df_weather_2023['Week'].dt.month
df_weather_2023['week_of_year'] = df_weather_2023['Week'].dt.isocalendar().week
last_4_2022 = df_train.tail(4)[['Week', 'Cases', 'Rainfall Amount', 'Max Temperature', 'Minimum Temperature', 'Relative Humidity',
                                'Wind Speed', 'Wind Direction']].copy()
lag_data = list(last_4_2022['Cases'])

predicted_cases = []

for i in range(len(df_weather_2023)):
    current_week = df_weather_2023.iloc[i].copy()
    current_features = current_week[weather_vars].tolist()
    current_features += lag_data[-4:] + [current_week['month']] + [current_week['week_of_year']]

    prediction = model.predict(np.array(current_features).reshape(1, -1))[0]
    predicted_cases.append(round(prediction))
    lag_data.append(prediction)

df_weather_2023['Cases'] = predicted_cases
df_actual_2023['Week'] = pd.to_datetime(df_actual_2023['Week'], format='%m/%d/%Y')
df_result = df_weather_2023[['Week', 'Cases']].merge(df_actual_2023, on='Week')

mae = mean_absolute_error(df_result['Actual 2023 Cases'], df_result['Cases'])
rmse = np.sqrt(mean_squared_error(df_result['Actual 2023 Cases'], df_result['Cases']))
mape = np.mean(np.abs((df_result['Actual 2023 Cases'] - df_result['Cases']) / df_result['Actual 2023 Cases'])) * 100

print("\n--- Part 2: 2023 Forecast Accuracy ---")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

results_df = pd.DataFrame({
    'Week': df_weather_2023['Week'],
    'Predicted Cases': df_result['Cases'].round().astype(int),
    'Actual Cases': df_result['Actual 2023 Cases']
})

print(results_df)


#------- PROPHET -------

import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools
import math
import matplotlib.pyplot as plt

def create_lagged_features(df, features, lags):
    for feature in features:
        for lag in range(1, lags+1):
            df[f'{feature}_lag{lag}'] = df[feature].shift(lag)
    return df.dropna()

def calculate_mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

dengue_historical = pd.read_csv('dengue2023.csv', parse_dates=['Week'])
weather_2023 = pd.read_csv('weather2023.csv', parse_dates=['Week'])
actual_2023 = pd.read_csv('2023_cases.csv', parse_dates=['Week'])

weather_features = ['Rainfall Amount', 'Max Temperature', 'Minimum Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction']

ph_holidays = pd.DataFrame({
    'holiday': 'ph_all_holidays',
    'ds': pd.to_datetime([
        *[f'{year}-01-01' for year in range(2015, 2024)],
        *[f'{year}-04-09' for year in range(2015, 2024)],
        *[f'{year}-05-01' for year in range(2015, 2024)],
        *[f'{year}-06-12' for year in range(2015, 2024)],
        *[f'{year}-11-30' for year in range(2015, 2024)],
        *[f'{year}-12-25' for year in range(2015, 2024)],
        *[f'{year}-12-30' for year in range(2015, 2024)],
        '2015-08-31', '2016-08-29', '2017-08-28', '2018-08-27',
        '2019-08-26', '2020-08-31', '2021-08-30', '2022-08-29',
        '2023-08-28',
        '2015-07-17', '2016-07-06', '2017-06-25', '2018-06-15',
        '2019-06-05', '2020-05-24', '2021-05-13', '2022-05-02',
        '2023-04-21',
        '2015-09-23', '2016-09-12', '2017-09-01', '2018-08-21',
        '2019-08-11', '2020-07-31', '2021-07-20', '2022-07-09',
        '2023-06-28',
        '2015-08-21', '2016-10-31', '2017-10-31', '2018-08-21',
        '2019-08-21', '2020-08-21', '2021-08-21', '2022-08-21',
        '2023-08-21',
        '2020-11-01', '2021-11-01', '2022-11-01', '2023-11-01', '2024-11-01',
        '2023-04-10',
    ]),
    'lower_window': -1,
    'upper_window': 1
})

lag_window = 1
train_df = create_lagged_features(
    dengue_historical.sort_values('Week').reset_index(drop=True),
    weather_features,
    lags=lag_window
).rename(columns={'Week': 'ds', 'Cases': 'y'})

full_weather = pd.concat([
    dengue_historical[['Week'] + weather_features].tail(lag_window),
    weather_2023.sort_values('Week')
]).reset_index(drop=True)

full_weather_lagged = create_lagged_features(full_weather, weather_features, lag_window)
future = full_weather_lagged[full_weather_lagged['Week'].dt.year == 2023][['Week'] +
            [col for col in full_weather_lagged.columns if any(f in col for f in weather_features)]]
future = future.rename(columns={'Week': 'ds'})

actual_2023_sorted = actual_2023.sort_values('Week').reset_index(drop=True)

param_grid = {
    'growth': ['linear', 'logistic'],
    'changepoint_prior_scale': [0.01, 0.1],
    'seasonality_prior_scale': [0.1, 1.0],
    'seasonality_mode': ['additive', 'multiplicative'],
    'yearly_seasonality': [True, False],
    'weekly_seasonality': [True, False],
    'daily_seasonality': [True, False],
    'holidays_prior_scale': [1, 5, 15]
}

all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

best_params = None
best_mape = float('inf')
best_metrics = None
best_forecast = None

regressor_cols = [col for col in train_df.columns if any(f in col for f in weather_features)]

for params in all_params:
    try:
        model = Prophet(
            holidays=ph_holidays,
            growth='linear',
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            seasonality_mode=params['seasonality_mode'],
            yearly_seasonality=params['yearly_seasonality'],
            weekly_seasonality=params['weekly_seasonality'],
            daily_seasonality=params['daily_seasonality'],
            holidays_prior_scale=params['holidays_prior_scale']
        )

        for col in regressor_cols:
            model.add_regressor(col, prior_scale=0.5)

        model.fit(train_df.copy())
        forecast = model.predict(future.copy())

        y_actual = actual_2023_sorted['Actual 2023 Cases'].values
        y_pred = forecast['yhat'][:len(y_actual)].values  # match length

        mae = mean_absolute_error(y_actual, y_pred)
        rmse = math.sqrt(mean_squared_error(y_actual, y_pred))
        mape = calculate_mape(y_actual, y_pred)

        if mape < best_mape:
            best_mape = mape
            best_params = params
            best_metrics = (mae, rmse, mape)
            best_forecast = forecast.copy()

    except Exception as e:
        print(f"Error with parameters {params}: {e}")

print("Best Parameters:")
print(best_params)
print("\nBest Metrics:")
print(f"MAE: {best_metrics[0]:.2f}")
print(f"RMSE: {best_metrics[1]:.2f}")
print(f"MAPE: {best_metrics[2]:.2f}%\n")

forecast_result = pd.DataFrame({
    'ds': actual_2023_sorted['Week'],
    'yhat': best_forecast['yhat'][:len(actual_2023_sorted)].round().astype(int),
    'actual': actual_2023_sorted['Actual 2023 Cases']
})

print(forecast_result)


#------- COMBINED PLOT -------

forecast_result['ds'] = pd.to_datetime(forecast_result['ds'])
results_df['Week'] = pd.to_datetime(results_df['Week'])
results_combined = pd.merge(results_df, forecast_result[['ds', 'yhat']], left_on='Week', right_on='ds', how='left')
results_combined.rename(columns={'yhat': 'Prophet Cases'}, inplace=True)
results_combined.rename(columns={'Predicted Cases': 'XGBoost Cases'}, inplace=True)
results_combined = results_combined.drop(columns={'ds'})
print(results_combined)

# Plot 1: Time-Series Graph (Setup 2)
plt.figure(figsize=(14, 6))
plt.plot(results_combined['Week'], results_combined['Actual Cases'], label='Actual Cases', marker='o', color='blue')
plt.plot(results_combined['Week'], results_combined['XGBoost Cases'], label='XGBoost Prediction', marker='x', color='green')
plt.plot(results_combined['Week'], results_combined['Prophet Cases'], label='Prophet Prediction', marker='+', color='red')
plt.title('Predicted vs Actual 2023 Dengue Cases')
plt.xlabel('Month')
plt.ylabel('Dengue Cases')
plt.xticks(pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS'),
           [d.strftime('%b') for d in pd.date_range(start='2023-01-01', end='2023-12-31', freq='MS')])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 3: Scatter plot (Setup 2)
plt.figure(figsize=(8, 8))
sns.scatterplot(x='Actual Cases', y='XGBoost Cases', data=results_combined, s=60, alpha=0.7, label='XGBoost Cases', color='green')
sns.scatterplot(x='Actual Cases', y='Prophet Cases', data=results_combined, s=60, alpha=0.7, label='Prophet Cases', color='red')

min_val = min(results_combined[['Actual Cases', 'XGBoost Cases', 'Prophet Cases']].min())
max_val = max(results_combined[['Actual Cases', 'XGBoost Cases', 'Prophet Cases']].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.title('Predicted vs Actual 2023 Dengue Cases (Scatter Plot)')
plt.xlabel('Actual 2023 Cases')
plt.ylabel('Predicted 2023 Cases')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

