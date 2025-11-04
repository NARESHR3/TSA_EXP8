# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### NAME:NARESH.R
### REG_NO:212223240104

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing

# Load your data
df_full = pd.read_csv('tsa.csv')

# Fix column names if needed
df_full.columns = [col.lower() for col in df_full.columns]

# Parse and fix datetime column
if 'date' in df_full.columns:
    df_full['date'] = pd.to_datetime(df_full['date'], dayfirst=True, errors='coerce')
else:
    raise ValueError("No 'date' column found in tsa.csv!")

df_full = df_full.set_index('date')

# Select 'open' column and drop NA
df = df_full[['open']].dropna().astype(float)
df.rename(columns={'open': 'Open'}, inplace=True)

# Monthly aggregation (for time series)
data_monthly = df['Open'].resample('MS').mean()
data_monthly = data_monthly.ffill()  # Forward fill missing values

data = pd.DataFrame({'Open': data_monthly})
data.index.freq = 'MS'  # Set frequency explicitly to avoid statsmodels warning

print("Shape of the filtered dataset:", data.shape)
print("FIRST 10 ROWS of the dataset:")
print(data.head(10))
print("\n")

data['MA_5'] = data['Open'].rolling(window=5).mean()
data['MA_10'] = data['Open'].rolling(window=10).mean()

print("ROLLING MEAN (WINDOW 5) - FIRST 10 ROWS:")
print(data.head(10))
print("\n")

print("ROLLING MEAN (WINDOW 10) - FIRST 20 ROWS:")
print(data.head(20))
print("\n")

print("Displaying Moving Average Plot...")
plt.figure(figsize=(12, 6))
plt.plot(data['Open'].iloc[-200:], label='Original Open price')
plt.plot(data['MA_5'].iloc[-200:], label='Moving Average (Window=5)', color='orange', linestyle='--')
plt.plot(data['MA_10'].iloc[-200:], label='Moving Average (Window=10)', color='red', linestyle='--')
plt.title('Moving Average Model (Last 200 Months)')
plt.xlabel('Date')
plt.ylabel('Open Value')
plt.legend()
plt.grid(True)
plt.show()

print("Displaying Exponential Smoothing Plot...")
model_es = SimpleExpSmoothing(data['Open'])
fit_es = model_es.fit(smoothing_level=0.2, optimized=False)  
data['ExpSmoothing'] = fit_es.fittedvalues

plt.figure(figsize=(12, 6))
plt.plot(data['Open'].iloc[-200:], label='Original Open price')
plt.plot(data['ExpSmoothing'].iloc[-200:], label='Exponential Smoothing (alpha=0.2)', color='green', linestyle='--')
plt.title('Simple Exponential Smoothing (Last 200 Months)')
plt.xlabel('Date')
plt.ylabel('Open Value')
plt.legend()
plt.grid(True)
plt.show()

print("Experiment Complete.")

```
### OUTPUT  :

### Plot Transform Dataset :

# FIRST 10 ROWS:
<img width="501" height="666" alt="image" src="https://github.com/user-attachments/assets/45b47f10-ef0f-4334-a83d-035f396b42d4" />


# FIRST 20 ROWS:

<img width="479" height="515" alt="image" src="https://github.com/user-attachments/assets/54a7e19e-57ab-41a6-a606-5877095ebec0" />

### Moving Average :
<img width="1235" height="665" alt="image" src="https://github.com/user-attachments/assets/23014589-c943-402f-bc9b-6ed2ab027027" />


### Exponential Smoothing :
<img width="1112" height="610" alt="image" src="https://github.com/user-attachments/assets/bbee8043-f35a-4cb3-9709-4fd2306b2a2e" />




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
