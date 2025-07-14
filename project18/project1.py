
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

# cell 7
# 🧼 4. Quick preprocessing (for the example)
# Replace -999 with NaN
train.replace(-999, np.nan, inplace=True)
test.replace(-999, np.nan, inplace=True)



# cell 8
# Drop rows with missing values (only for demo purposes – improve this in your model!)
train_clean = train.dropna()

# 🧠 5. Simple model: Linear Regression
features = ['WS2M', 'T2M', 'T2MWET', 'T2MDEW', 'RH2M', 'PS', 'QV2M']
X = train_clean[features]
y = train_clean['Target']



# cell 9
from sklearn.metrics import mean_squared_error
# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"✅ Validation RMSE: {rmse:.4f}")



# cell 10
# 📤 6. Predict on test set and create submission file
X_test = test[features].copy()
X_test.fillna(X.mean(), inplace=True)  # simple fill for the demo – use better imputation in real solutions

# Make predictions
test_predictions = model.predict(X_test)



# cell 11
# Create submission file
submission = test[['ID']].copy()
submission['Target'] = test_predictions
submission.to_csv("SampleSubmission.csv", index=False)

print("🎉 SampleSubmission.csv created!")
submission.head()


