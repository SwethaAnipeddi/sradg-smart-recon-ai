import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Load your historical reconciliation data
# Assuming you have a CSV file with relevant reconciliation data
df = pd.read_csv('reconciliation_data.csv')

# Step 2: Feature engineering
# For this example, we are using 'amount' and 'date' as features.
# You can modify this according to the features available in your dataset.

# Convert 'date' to numerical features (if needed)
df['date'] = pd.to_datetime(df['date'])
df['date_numeric'] = df['date'].apply(lambda x: x.toordinal())

# Features (add or modify columns based on your dataset)
features = ['amount', 'date_numeric']

# Extracting feature columns
X = df[features]

# Step 3: Normalize data (important for some models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Model Training - Use Isolation Forest to detect anomalies
model = IsolationForest(contamination=0.01)  # 'contamination' is the proportion of outliers
model.fit(X_scaled)

# Step 5: Predict anomalies
df['anomaly_score'] = model.decision_function(X_scaled)
df['anomaly'] = model.predict(X_scaled)

# Anomalies are labeled as -1, non-anomalies as 1
df['anomaly'] = df['anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Step 6: Analyze and flag anomalies in the reconciliation process
print(df[['date', 'amount', 'anomaly']].head(20))  # Print the flagged anomalies for inspection

# Step 7: Save the results if needed
df.to_csv('reconciliation_anomalies.csv', index=False)
