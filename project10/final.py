import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pykalman import KalmanFilter
import joblib

import warnings
warnings.filterwarnings('ignore')


# Load the data
train_data = pd.read_csv('Train.csv')
test_data = pd.read_csv('Test.csv')
sample_submission = pd.read_csv('SampleSubmission.csv')

# Parse dates from IDs
train_data['parsed_date'] = pd.to_datetime(train_data['ID'].apply(lambda x: x.split('_')[1]))
test_data['parsed_date'] = pd.to_datetime(test_data['ID'].apply(lambda x: x.split('_')[1]))

# Extract numeric IDs
train_data['area_id'] = pd.to_numeric(train_data['ID'].apply(lambda x: x.split('_')[0]))
test_data['area_id'] = pd.to_numeric(test_data['ID'].apply(lambda x: x.split('_')[0]))

# Model configurations
rmse_values = []
arima_order = (2, 0, 0)
seasonal_order = (2, 0, 1, 12)

# Store predictions
prediction_list = []

# Loop through each unique area ID
for area in train_data['area_id'].unique():
    # Filter the data for the current area
    area_train_data = train_data.loc[train_data['area_id'] == area]
    area_test_data = test_data.loc[test_data['area_id'] == area]
    
    try:
        # Fit SARIMA model
        sarimax_model = SARIMAX(area_train_data['burn_area'],
                                order=arima_order,
                                seasonal_order=seasonal_order)
        sarimax_fit = sarimax_model.fit()

        # Forecast future values
        forecast_horizon = len(area_test_data)
        sarimax_forecast = sarimax_fit.get_forecast(steps=forecast_horizon, freq='MS')
        forecasted_values = sarimax_forecast.predicted_mean

        # Apply Kalman Filter
        kalman_filter = KalmanFilter(initial_state_mean=forecasted_values.iloc[0], n_dim_obs=1)
        kalman_smoothed_values, _ = kalman_filter.filter(forecasted_values.values)
        kalman_forecast = pd.Series(kalman_smoothed_values.flatten(), index=area_test_data.index, name='burn_area')

        # Store predictions
        area_test_data.set_index('parsed_date', inplace=True)
        prediction_data = pd.DataFrame({'ID': str(area) + '_' + area_test_data.index.strftime('%Y-%m-%d').map(str),
                                        'burn_area': kalman_forecast.values})

    except Exception as error:
        print(f"Failed to model area {area}: {error}. Using ARIMA model.")
        try:
            arima_model = auto_arima(area_train_data['burn_area'], m=12, seasonal=True,
                                     start_p=0, start_q=0, max_order=5, test='adf', error_action='ignore',
                                     suppress_warnings=True, stepwise=True, trace=False)
            forecast_len = len(area_test_data)
            arima_forecast = arima_model.predict(n_periods=forecast_len)

            kalman_filter = KalmanFilter(initial_state_mean=arima_forecast[0], n_dim_obs=1)
            kalman_smoothed_values, _ = kalman_filter.filter(arima_forecast)
            kalman_forecast = pd.Series(kalman_smoothed_values.flatten(), index=area_test_data.index, name='burn_area')
            area_test_data.set_index('parsed_date', inplace=True)
            prediction_data = pd.DataFrame({'ID': str(area) + '_' + area_test_data.index.strftime('%Y-%m-%d').map(str),
                                            'burn_area': kalman_forecast.values})
        except Exception as fallback_error:
            print(f"ARIMA model failed for area {area}: {fallback_error}. Falling back to zero prediction.")
            area_test_data.set_index('parsed_date', inplace=True)
            prediction_data = pd.DataFrame({'ID': str(area) + '_' + area_test_data.index.strftime('%Y-%m-%d').map(str),
                                            'burn_area': kalman_forecast.values})
    
    # Add predictions to the list
    prediction_list.append(prediction_data)
    
# Combine predictions for all areas
final_predictions = pd.DataFrame(columns=['ID', 'burn_area'])

# Loop through each time step (48 months)
for step in range(48):
    step_predictions = pd.DataFrame(columns=['ID', 'burn_area'])

    # Gather predictions for all areas at each step
    for idx in range(len(prediction_list)):
        step_predictions = pd.concat([step_predictions, prediction_list[idx].loc[prediction_list[step].index == step]], ignore_index=True)
    
    final_predictions = pd.concat([final_predictions, step_predictions], ignore_index=True)

# Clip predictions and prepare submission file
sample_submission['burn_area'] = final_predictions['burn_area'].clip(0, 1)

# Save submission
sample_submission.to_csv('submit.csv', index=False)

