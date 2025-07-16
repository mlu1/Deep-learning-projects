# cell 0
# 🌧️ Zindi Hackathon - Starter Notebook: Predicting Corrected Precipitation (PRECTOTCORR)

# 📥 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# cell 1
# 📁 2. Load the datasets
train = pd.read_csv("Train_data.csv")
test = pd.read_csv("Test_data.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
train.head()


# cell 2
train.head()



# cell 3
test.head()


# cell 4
train.shape[0], train['ID'].nunique(), test.shape[0], test['ID'].nunique()


# cell 5
## Are the ID unique, no interaction between train & test
train['ID'].isin(test['ID']).sum(), test['ID'].isin(train['ID']).sum()


# cell 6
# 📊 3. Quick visualization

# Define basic features for visualization
basic_features = ['WS2M', 'T2M', 'T2MWET', 'T2MDEW', 'RH2M', 'PS', 'QV2M']

# Visualize the distribution of the target variable
plt.figure(figsize=(10, 6))
plt.hist(train['Target'], bins=30, alpha=0.7)
plt.title('Distribution of Target (Precipitation)')
plt.xlabel('Target Value')
plt.ylabel('Frequency')
plt.axvline(train['Target'].mean(), color='red', linestyle='--', label=f'Mean: {train["Target"].mean():.2f}')
plt.axvline(train['Target'].median(), color='green', linestyle='--', label=f'Median: {train["Target"].median():.2f}')
plt.legend()
plt.show()

# Correlation heatmap of original features
plt.figure(figsize=(12, 10))
correlation = train[basic_features + ['Target']].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

# cell 7
# 🧼 4. Quick preprocessing (for the example)
# Replace -999 with NaN
train.replace(-999, np.nan, inplace=True)
test.replace(-999, np.nan, inplace=True)



# cell 8
# Drop rows with missing values (only for demo purposes – improve this in your model!)
train_clean = train.dropna()

# Define features before using them in feature engineering
features = ['WS2M', 'T2M', 'T2MWET', 'T2MDEW', 'RH2M', 'PS', 'QV2M']

# 🔧 Feature Engineering
# 1. Create date-related features
def add_date_features(df):
    # Convert to datetime
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    # Extract basic date components
    df['day_of_year'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['day'] = df['DATE'].dt.day
    
    # Create season feature (1=spring, 2=summer, 3=fall, 4=winter)
    df['season'] = df['month'].apply(lambda x: (x%12 + 3)//3)
    
    # Create cyclical features for month and day of year
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    return df

# 2. Create weather-specific features
def add_weather_features(df):
    # Temperature-humidity index

    
    es = 6.112 * np.exp((17.67 * df['T2M']) / (df['T2M'] + 243.5))          # hPa
    ea = es * df['RH2M'] / 100
    df['saturation_deficit'] = es - ea                                         # VPD proxy
    df['theta_e'] = (df['T2M'] + 273.15) * (1000 / train['PS'])**0.286 * np.exp((3376/df['T2M']) - 2.54)
    df['lcl_est'] = 125*(df['T2M'] - df['T2MDEW'])                       # lifting‑condensation‑level height (m)
    


    df['temp_humidity_index'] = 0.8 * df['T2M'] + df['RH2M'] / 100 * (df['T2M'] - 14.4) + 46.4
    
    # Dew point depression (difference between temp and dew point)
    df['dew_point_depression'] = df['T2M'] - df['T2MDEW']
    
    # Wind chill approximation
    df['wind_chill'] = 13.12 + 0.6215 * df['T2M'] - 11.37 * df['WS2M']**0.16 + 0.3965 * df['T2M'] * df['WS2M']**0.16
    
    # Humidity and temperature interaction
    df['humidity_temp'] = df['RH2M'] * df['T2M']
    
    # Pressure and temperature interaction
    df['pressure_temp'] = df['PS'] * df['T2M']
    
    # Wind and humidity interaction
    df['wind_humidity'] = df['WS2M'] * df['RH2M']
 
    df['heat_index'] = -42.379 + 2.04901523*df['T2M'] + 10.14333127*df['RH2M']/100 - 0.22475541*df['T2M']*df['RH2M']/100
    
    # Vapor pressure deficit (VPD) - relates to evaporation potential
    saturated_vapor_pressure = 6.112 * np.exp((17.67 * df['T2M']) / (df['T2M'] + 243.5))
    actual_vapor_pressure = saturated_vapor_pressure * (df['RH2M'] / 100)
    df['vapor_pressure_deficit'] = saturated_vapor_pressure - actual_vapor_pressure
    
    # Potential evapotranspiration approximation
    df['potential_evapotranspiration'] = 0.0023 * (df['T2M'] + 17.8) * np.sqrt(df['T2M'] - df['T2MDEW']) * df['PS'] / 100
    
    # Convective Available Potential Energy (CAPE) proxy - simplified using available features
    df['convective_energy_proxy'] = df['T2M'] * df['vapor_pressure_deficit'] * df['WS2M']
    
    # Wet bulb depression (diff between temp and wet bulb temp) - indicates evaporative cooling potential
    df['wet_bulb_depression'] = df['T2M'] - df['T2MWET']
    
    # Precipitable water proxy (using vapor pressure)
    df['precipitable_water_proxy'] = df['QV2M'] * np.log(df['PS'])
    
    # Instability index (difference between surface and upper air temperature proxies)
    df['instability_index'] = df['T2M'] - df['T2MWET'] - (df['PS'] - 950) / 100


    df['cape_proxy']   = df['theta_e'] * df['WS2M']
    df['k_index']      = (df['T2M'] - df['T2MWET']) + train['RH2M']/10
    df['showalter']    = (df['T2MDEW'] - (df['T2M'] - 6))                # crude mid‑level lapse proxy


    
    return df

# 3. Create interaction and polynomial features
def add_polynomial_features(df, features, degree=2):
    # Interaction terms
    for i, feat1 in enumerate(features):
        for feat2 in features[i+1:]:
            df[f'{feat1}_{feat2}_interact'] = df[feat1] * df[feat2]
            
        # Polynomial terms
        if degree >= 2:
            df[f'{feat1}_squared'] = df[feat1] ** 2
        
        if degree >= 3:
            df[f'{feat1}_cubed'] = df[feat1] ** 3
            
    return df



def add_clustering_features(df, n_clusters=5):
    """Add cluster-based features to identify weather patterns"""
    try:
        # Select features for clustering
        cluster_features = ['T2M', 'RH2M', 'WS2M', 'PS']
        
        # Check if all required features exist
        missing_features = [feat for feat in cluster_features if feat not in df.columns]
        if missing_features:
            print(f"  Warning: Missing required features for clustering: {missing_features}")
            return df
        
        # Scale the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[cluster_features])
        
        # Perform K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['weather_cluster'] = kmeans.fit_predict(scaled_features)
        
        # Create one-hot encoded cluster features
        for i in range(n_clusters):
            df[f'weather_cluster_{i}'] = (df['weather_cluster'] == i).astype(int)
    
    except Exception as e:
        print(f"  Error in clustering: {e}")
        print("  Continuing without clustering features...")
    
    return df



# 6. Frequency domain features - to capture oscillatory patterns in weather data
def add_frequency_features(df, target_cols=['T2M', 'RH2M', 'WS2M']):
    """
    Extract frequency domain features using Fast Fourier Transform (FFT)
    to identify potential periodic patterns in weather data.
    """
    print("- Computing frequency domain features...")
    
    # Need at least 100 samples for meaningful FFT
    if len(df) < 100:
        print("  Not enough data for frequency analysis, skipping.")
        return df
    
    # Get the top 3 frequency components for each column
    for col in target_cols:
        if col in df.columns:
            try:
                # Get values and replace NaN with mean
                values = df[col].values
                values = np.nan_to_num(values, nan=np.nanmean(values))
                
                # Compute FFT
                fft_values = np.fft.rfft(values)
                fft_magnitudes = np.abs(fft_values)
                
                # Make sure we have enough frequency components
                if len(fft_magnitudes) <= 1:
                    print(f"  Warning: Not enough frequency components for {col}, skipping.")
                    continue
                    
                # Get indices of top 3 frequency components (excluding DC component at index 0)
                # Ensure we don't exceed array bounds
                max_components = min(3, len(fft_magnitudes) - 1)
                if max_components <= 0:
                    continue
                    
                top_indices = np.argsort(fft_magnitudes[1:])[-max_components:] + 1
                
                # Store magnitude of top frequency components
                for i, idx in enumerate(top_indices):
                    df[f'{col}_fft_mag_{i+1}'] = fft_magnitudes[idx]
            except Exception as e:
                print(f"  Error computing FFT for {col}: {e}")
                continue
    
    return df


def add_decomposition_features(train_df, test_df, n_components=10):
    """
    Apply PCA to reduce dimensionality and create uncorrelated features.
    This is useful for handling multicollinearity in the engineered features.
    
    Parameters:
    -----------
    train_df, test_df : DataFrames with features
    n_components : Number of PCA components to retain
    
    Returns:
    --------
    train_df, test_df with PCA features added
    """
    try:
        # Only apply to numerical columns
        numerical_cols = [col for col in train_df.columns 
                         if col != 'DATE' and np.issubdtype(train_df[col].dtype, np.number)
                         and not col.startswith('weather_cluster')  # Skip one-hot encoded columns
                         and not pd.isna(train_df[col]).any()       # Skip columns with NaN
                         and not pd.isna(test_df[col]).any()        # Skip columns with NaN
                         and col in test_df.columns]                # Make sure column exists in test data
        
        if len(numerical_cols) < 2:
            print("  Not enough valid numerical features for PCA. Skipping decomposition.")
            return train_df, test_df
        
        if len(numerical_cols) < n_components:
            print(f"  Not enough numerical features ({len(numerical_cols)}) for {n_components} PCA components.")
            n_components = max(len(numerical_cols) // 2, 2)  # Use at least 2 components if possible
            print(f"  Reducing to {n_components} components.")
        
        # Combine data for PCA fit
        combined = pd.concat([train_df[numerical_cols], test_df[numerical_cols]])
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(combined)
        
        # Transform each dataset
        train_pca = pca.transform(train_df[numerical_cols])
        test_pca = pca.transform(test_df[numerical_cols])
        
        # Add components as new features
        for i in range(n_components):
            train_df[f'pca_comp_{i+1}'] = train_pca[:, i]
            test_df[f'pca_comp_{i+1}'] = test_pca[:, i]
        
        # Print explained variance
        explained_var = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        print(f"  PCA components explain {cumulative_var[-1]*100:.2f}% of variance")
        print(f"  Top 3 components explain: {explained_var[:min(3, len(explained_var))]*100}")
        
    except Exception as e:
        print(f"  Error in PCA decomposition: {e}")
        print("  Continuing without PCA features...")
    
    return train_df, test_df


def add_temporal_features(df, features_to_lag, window_sizes=[3, 7]):
    """
    Adds lag and rolling window features for specified columns.
    This helps the model understand trends and seasonality.
    """
    print("- Adding temporal (lag/rolling) features...")
    
    # Ensure data is sorted by date to correctly calculate lags and rolling windows
    df = df.sort_values('DATE').reset_index(drop=True)
    
    for col in features_to_lag:
        if col in df.columns:
            for window in window_sizes:
                # Lag features (value from 'window' days ago)
                df[f'{col}_lag_{window}'] = df[col].shift(window)
                
                # Rolling window features (stats over the last 'window' days)
                # shift(1) is used to ensure we only use past data for the rolling window
                rolling_window = df[col].shift(1).rolling(window=window)
                df[f'{col}_rolling_mean_{window}'] = rolling_window.mean()
                df[f'{col}_rolling_std_{window}'] = rolling_window.std()
                df[f'{col}_rolling_max_{window}'] = rolling_window.max()
                df[f'{col}_rolling_min_{window}'] = rolling_window.min()

    # Fill NaNs created by lag/rolling features.
    # Using backfill to avoid data leakage from future values in a real scenario.
    # For this competition, a simple fill is okay.
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True) # Fill any remaining NaNs at the end
    
    return df


# Apply feature engineering to training and test data
print("Applying feature engineering...")
train_clean = add_date_features(train_clean)
train_clean = add_weather_features(train_clean)
train_clean = add_polynomial_features(train_clean, features)
print("- Adding frequency domain features...")
train_clean = add_frequency_features(train_clean)
train_clean = add_temporal_features(train_clean, features) # Add temporal features



# Now for test data (handle missing values before feature engineering)
test_fe = test.copy()
test_fe.fillna(train_clean[features].mean(), inplace=True)  # better imputation than just using mean
test_fe = add_date_features(test_fe)
test_fe = add_weather_features(test_fe)
test_fe = add_polynomial_features(test_fe, features)
test_fe = add_frequency_features(test_fe)
test_fe = add_temporal_features(test_fe, features) # Add temporal features


print("- Adding PCA decomposition features...")
train_clean, test_fe = add_decomposition_features(train_clean, test_fe, n_components=10)

def add_fft_energy(df, col):
    vals = np.nan_to_num(df[col].values, nan=np.nanmean(df[col]))
    fft_vals = np.fft.rfft(vals)
    power = (np.abs(fft_vals)**2)
    df[f'{col}_low_freq_energy']  = power[1:5].sum() / power.sum()
    df[f'{col}_daily_cycle_phase'] = np.angle(fft_vals[1])      # first harmonic phase
    return df

train_clean = add_fft_energy(train_clean, 'T2M')
test_fe = add_fft_energy(test_fe, 'T2M')


# Get all engineered feature columns (excluding original features if desired)
original_features = features.copy()
engineered_features = [col for col in train_clean.columns if col not in train.columns and col != 'DATE']
all_features = original_features + engineered_features

print(f"Original features: {len(original_features)}")
print(f"Engineered features: {len(engineered_features)}")
print(f"Total features: {len(all_features)}")

# Create X and y with all features
X = train_clean[all_features]
y = train_clean['Target']


model = LinearRegression()
model.fit(X, y)

# Evaluate the model

from sklearn.feature_selection import SelectFromModel

# Create a selector based on feature importance
selector = SelectFromModel(model, threshold="median")
selector.fit(X, y)

# Get the selected features

selected_features_mask = selector.get_support()
selected_features = [feature for feature, selected in zip(all_features, selected_features_mask) if selected]
print(f"Selected {len(selected_features)} out of {len(all_features)} features")
print("Top selected features:", selected_features[:10])  # Show first 10

# Train final model with selected features
X_selected = X[selected_features]

print(len(X_selected.columns))

X_test_selected = test_fe[selected_features].copy()


# cell 9
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor
# Split into train and validation sets


tscv = TimeSeriesSplit(n_splits=6)          # folds respect chronology
oof, preds = np.zeros(len(y)), []

for train_idx, val_idx in tscv.split(X_selected):
    X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    
    gbm = LGBMRegressor(
        random_state=42
    )
    gbm.fit(X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            callbacks=[])

    oof[val_idx] = gbm.predict(X_val)
    preds.append(gbm.predict(X_test_selected))

print("OOF RMSE:", np.sqrt(mean_squared_error(y, oof)))


# --- PREDICT TEST DATA ------------------------------------------
# 1. Choose the feature matrix you trained on
TEST_X =  X_test_selected            # or   test_fe[selected_features]

# 2. Predict with the trained model (replace gbm with your model variable)
test_predictions = gbm.predict(TEST_X)     # or np.mean(preds, axis=0) if you ensembled

# 3. Post‑process (rainfall can’t be negative)
test_predictions = np.clip(test_predictions, 0, None)

# 4. Create submission CSV
submission = pd.DataFrame({
    'ID': test['ID'],
    'Target': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("✅ submission.csv written   →   rows:", len(submission))

