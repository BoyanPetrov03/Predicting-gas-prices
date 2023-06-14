import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# Load the natural gas price data from the local file
data = pd.read_csv('daily_csv.csv')

# Assuming your data has two columns: 'Date' and 'Price'
# Make sure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Sort the data in ascending order of date
data.sort_index(inplace=True)

# Resample the data to daily frequency (D)
data = data.resample('D').mean()

# Split the data into training and testing sets
train_data = data.loc[:'2020-08-31']
test_data = data.loc['2020-09-01':]

# Check if test_data is empty
if test_data.empty:
    print("No test data available.")
else:
    # Create and fit the ARIMA model
    model = ARIMA(train_data['Price'], order=(1, 1, 1))  # Adjust the order values as needed
    model_fit = model.fit()

    # Generate predictions for the test data
    start_date = test_data.index[0]  # Start date of the test data
    end_date = test_data.index[-1]  # End date of the test data
    forecast = model_fit.predict(start=start_date, end=end_date)

    # Print the predicted prices
    print("forecast")
    print(forecast)

    # Evaluate the model performance (optional)
    actual_prices = test_data['Price'].values
    valid_predictions = forecast
    mse = np.mean((valid_predictions - actual_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(valid_predictions - actual_prices))
    print("RMSE:", rmse)
    print("MAE:", mae)

