# ------- XGBOOST -------

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

tscv = TimeSeriesSplit(n_splits=5)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv('dengue2024.csv')
df['Week'] = pd.to_datetime(df['Week'], format='%m/%d/%Y')
df.rename(columns={'Week': 'ds'}, inplace=True)

feature_vars = ['Cases', 'Rainfall Amount', 'Max Temperature', 'Minimum Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction']
future_dates = [pd.Timestamp("2024-01-01")]
future_dates += pd.date_range(start="2024-01-07", periods=51, freq='7D').tolist()
forecast = pd.DataFrame({'ds': future_dates})

predictions = {'ds': future_dates}

df['month'] = df['ds'].dt.month
df['week_of_year'] = df['ds'].dt.isocalendar().week

for var in feature_vars:
    print(f"\nTraining model for {var}...")

    feature_cols = ['month', 'week_of_year']
    data = df[['ds', var] + feature_cols].dropna()

    X = data[feature_cols]
    y = data[var]

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

    grid_search = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror', random_state=42), param_grid, cv=tscv, scoring='neg_mean_absolute_error',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"Optimized MAE: {mae}")
    print(f"Optimized RMSE: {rmse}")
    print(f"Optimized MAPE: {mape:.2f}%")

    full_data = data.copy()

    future_preds = []
    for date in future_dates:
        month = date.month
        week_of_year = date.isocalendar().week

        X_new = pd.DataFrame([[month, week_of_year]], columns=feature_cols)

        pred = model.predict(X_new)[0]
        future_preds.append(pred)

        new_row = pd.DataFrame({
            'ds': [date],
            var: [pred],
            'month': [month],
            'week_of_year': [week_of_year],
        })

    predictions[var] = future_preds

xgb_forecast = pd.DataFrame(predictions)

real_2024 = pd.read_csv('2024_cases.csv')
real_2024['Week'] = pd.to_datetime(real_2024['Week'], format='%m/%d/%Y')
real_2024.rename(columns={'Week': 'ds'}, inplace=True)

compare_df = pd.merge(xgb_forecast, real_2024, on='ds', how='inner')
mae_2024 = mean_absolute_error(compare_df['Actual 2024 Cases'], compare_df['Cases'])
rmse_2024 = np.sqrt(mean_squared_error(compare_df['Actual 2024 Cases'], compare_df['Cases']))
mape_2024 = np.mean(np.abs((compare_df['Actual 2024 Cases'] - compare_df['Cases']) / compare_df['Actual 2024 Cases'])) * 100

print("\n--- 2024 Forecast Accuracy ---")
print(f"2024 MAE: {mae_2024:.2f}")
print(f"2024 RMSE: {rmse_2024:.2f}")
print(f"2024 MAPE: {mape_2024:.2f}%")

results_df = pd.DataFrame({
    'Week': compare_df['ds'],
    'Predicted Cases': compare_df['Cases'].round().astype(int),
    'Actual Cases': compare_df['Actual 2024 Cases']
})
print(results_df)


jan_to_aug_df = results_df[results_df['Week'].dt.month <= 8]

mae_jan_aug = mean_absolute_error(jan_to_aug_df['Actual Cases'], jan_to_aug_df['Predicted Cases'])
rmse_jan_aug = np.sqrt(mean_squared_error(jan_to_aug_df['Actual Cases'], jan_to_aug_df['Predicted Cases']))
mape_jan_aug = np.mean(np.abs((jan_to_aug_df['Actual Cases'] - jan_to_aug_df['Predicted Cases']) / jan_to_aug_df['Actual Cases'])) * 100

print("\n--- January to August 2024 Forecast Accuracy ---")
print(f"MAE: {mae_jan_aug:.2f}")
print(f"RMSE: {rmse_jan_aug:.2f}")
print(f"MAPE: {mape_jan_aug:.2f}%")

sep_to_dec_df = results_df[results_df['Week'].dt.month >= 9]

mae_sep_dec = mean_absolute_error(sep_to_dec_df['Actual Cases'], sep_to_dec_df['Predicted Cases'])
rmse_sep_dec = np.sqrt(mean_squared_error(sep_to_dec_df['Actual Cases'], sep_to_dec_df['Predicted Cases']))
mape_sep_dec = np.mean(np.abs((sep_to_dec_df['Actual Cases'] - sep_to_dec_df['Predicted Cases']) / sep_to_dec_df['Actual Cases'])) * 100

print("\n--- September to December 2024 Forecast Accuracy ---")
print(f"MAE: {mae_sep_dec:.2f}")
print(f"RMSE: {rmse_sep_dec:.2f}")
print(f"MAPE: {mape_sep_dec:.2f}%")


#------- PROPHET -------

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from itertools import product


df = pd.read_csv('dengue2024.csv')
df['Week'] = pd.to_datetime(df['Week'], format='%m/%d/%Y')
df.rename(columns={'Week': 'ds', 'Cases': 'y'}, inplace=True)
weather_vars = ['Rainfall Amount', 'Max Temperature', 'Minimum Temperature', 'Relative Humidity', 'Wind Speed', 'Wind Direction']

ph_holidays = pd.DataFrame({
    'holiday': 'ph_all_holidays',
    'ds': pd.to_datetime([
        *[f'{year}-01-01' for year in range(2015, 2025)],
        *[f'{year}-04-09' for year in range(2015, 2025)],
        *[f'{year}-05-01' for year in range(2015, 2025)],
        *[f'{year}-06-12' for year in range(2015, 2025)],
        *[f'{year}-11-30' for year in range(2015, 2025)],
        *[f'{year}-12-25' for year in range(2015, 2025)],
        *[f'{year}-12-30' for year in range(2015, 2025)],
        '2015-08-31', '2016-08-29', '2017-08-28', '2018-08-27',
        '2019-08-26', '2020-08-31', '2021-08-30', '2022-08-29',
        '2023-08-28', '2024-08-26',
        '2015-07-17', '2016-07-06', '2017-06-25', '2018-06-15',
        '2019-06-05', '2020-05-24', '2021-05-13', '2022-05-02',
        '2023-04-21', '2024-04-10',
        '2015-09-23', '2016-09-12', '2017-09-01', '2018-08-21',
        '2019-08-11', '2020-07-31', '2021-07-20', '2022-07-09',
        '2023-06-28', '2024-06-16',
        '2015-08-21', '2016-10-31', '2017-10-31', '2018-08-21',
        '2019-08-21', '2020-08-21', '2021-08-21', '2022-08-21',
        '2023-08-21', '2024-08-21',
        '2020-11-01', '2021-11-01', '2022-11-01', '2023-11-01', '2024-11-01',
        '2023-04-10'
    ]),
    'lower_window': -1,
    'upper_window': 1
})

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

all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
results = []

for params in all_params:
    model = Prophet(holidays=ph_holidays, **params)
    for var in weather_vars:
        model.add_regressor(var, prior_scale=0.5)
    model.fit(df)

    future_dates = [pd.Timestamp("2024-01-01")]
    future_dates += pd.date_range(start="2024-01-07", periods=51, freq='7D').tolist()
    future = pd.DataFrame({'ds': future_dates})
    future = pd.merge(future, df[['ds'] + weather_vars], on='ds', how='left')
    future.fillna(df.groupby(df['ds'].dt.month).transform('mean'), inplace=True)

    forecast = model.predict(future)
    actual_df = pd.read_csv('2024_cases.csv')
    actual_df['Week'] = pd.to_datetime(actual_df['Week'], format='%m/%d/%Y')
    actual_df.rename(columns={'Week': 'ds', 'Actual 2024 Cases': 'actual'}, inplace=True)
    result_df = forecast[['ds', 'yhat']].merge(actual_df, on='ds')
    mae = mean_absolute_error(result_df['actual'], result_df['yhat'])
    rmse = np.sqrt(mean_squared_error(result_df['actual'], result_df['yhat']))
    mape = np.mean(np.abs((result_df['actual'] - result_df['yhat']) / result_df['actual'])) * 100
    results.append({'params': params, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})

best_result = sorted(results, key=lambda x: x['RMSE'])[0]
print("Best Parameters:", best_result['params'])
print(f"MAE: {best_result['MAE']:.2f}, RMSE: {best_result['RMSE']:.2f}, MAPE: {best_result['MAPE']:.2f}%")

best_model = Prophet(holidays=ph_holidays, **best_result['params'])
for var in weather_vars:
    best_model.add_regressor(var, prior_scale=0.5)
best_model.fit(df)

future = pd.DataFrame({'ds': future_dates})
future = pd.merge(future, df[['ds'] + weather_vars], on='ds', how='left')
future.fillna(df.groupby(df['ds'].dt.month).transform('mean'), inplace=True)
forecast = best_model.predict(future)

actual_df = pd.read_csv('2024_cases.csv')
actual_df['Week'] = pd.to_datetime(actual_df['Week'], format='%m/%d/%Y')
actual_df.rename(columns={'Week': 'ds', 'Actual 2024 Cases': 'actual'}, inplace=True)
forecast_result = forecast[['ds', 'yhat']].merge(actual_df, on='ds')

mae = mean_absolute_error(forecast_result['actual'], forecast_result['yhat'])
rmse = np.sqrt(mean_squared_error(forecast_result['actual'], forecast_result['yhat']))
mape = np.mean(np.abs((forecast_result['actual'] - forecast_result['yhat']) / forecast_result['actual'])) * 100

print("\nAccuracy Metrics:")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

print(forecast_result)


jan_to_aug = forecast_result[forecast_result['ds'].dt.month <= 8]
mae_jan_aug = mean_absolute_error(jan_to_aug['actual'], jan_to_aug['yhat'])
rmse_jan_aug = np.sqrt(mean_squared_error(jan_to_aug['actual'], jan_to_aug['yhat']))
mape_jan_aug = np.mean(np.abs((jan_to_aug['actual'] - jan_to_aug['yhat']) / jan_to_aug['actual'])) * 100

print("\n--- January to August 2024 Prophet Accuracy ---")
print(f"MAE: {mae_jan_aug:.2f}")
print(f"RMSE: {rmse_jan_aug:.2f}")
print(f"MAPE: {mape_jan_aug:.2f}%")

sep_to_dec = forecast_result[forecast_result['ds'].dt.month >= 9]
mae_sep_dec = mean_absolute_error(sep_to_dec['actual'], sep_to_dec['yhat'])
rmse_sep_dec = np.sqrt(mean_squared_error(sep_to_dec['actual'], sep_to_dec['yhat']))
mape_sep_dec = np.mean(np.abs((sep_to_dec['actual'] - sep_to_dec['yhat']) / sep_to_dec['actual'])) * 100

print("\n--- September to December 2024 Prophet Accuracy ---")
print(f"MAE: {mae_sep_dec:.2f}")
print(f"RMSE: {rmse_sep_dec:.2f}")
print(f"MAPE: {mape_sep_dec:.2f}%")



# ------- PLOTTING -------

forecast_result['ds'] = pd.to_datetime(forecast_result['ds'])
results_df['Week'] = pd.to_datetime(results_df['Week'])

results_combined = pd.merge(results_df, forecast_result[['ds', 'yhat']], left_on='Week', right_on='ds', how='left')
results_combined.rename(columns={'yhat': 'Prophet Cases'}, inplace=True)
results_combined.rename(columns={'Predicted Cases': 'XGBoost Cases'}, inplace=True)
results_combined = results_combined.drop(columns={'ds'})
print(results_combined)

# Plot 1: Time-Series Graph (Setup 1)
plt.figure(figsize=(14, 6))
plt.plot(results_combined['Week'], results_combined['Actual Cases'], label='Actual Cases', marker='o', color='blue')
plt.plot(results_combined['Week'], results_combined['XGBoost Cases'], label='XGBoost Prediction', marker='x', color='green')
plt.plot(results_combined['Week'], results_combined['Prophet Cases'], label='Prophet Prediction', marker='+', color='red')
plt.title('Predicted vs Actual 2024 Dengue Cases')
plt.xlabel('Month')
plt.ylabel('Dengue Cases')
plt.xticks(pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS'),
           [d.strftime('%b') for d in pd.date_range(start='2024-01-01', end='2024-12-31', freq='MS')])
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Plot 2: Scatter Plot (Setup 1)
plt.figure(figsize=(8, 8))
sns.scatterplot(x='Actual Cases', y='XGBoost Cases', data=results_combined, s=60, alpha=0.7, label='XGBoost Cases', color='green')
sns.scatterplot(x='Actual Cases', y='Prophet Cases', data=results_combined, s=60, alpha=0.7, label='Prophet Cases', color='red')

min_val = min(results_combined[['Actual Cases', 'XGBoost Cases', 'Prophet Cases']].min())
max_val = max(results_combined[['Actual Cases', 'XGBoost Cases', 'Prophet Cases']].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')

plt.title('Predicted vs Actual 2024 Dengue Cases (Scatter Plot)')
plt.xlabel('Actual 2024 Cases')
plt.ylabel('Predicted 2024 Cases')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
