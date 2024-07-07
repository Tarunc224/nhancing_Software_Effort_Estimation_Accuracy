import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the COCOMO81 dataset
'''data = pd.read_csv('/Users/tarunchintada/Downloads/desharnais.csv')

# Assuming the dataset has columns for input  without feature selection and the target 'actual'
X = data.drop(columns=['Effort'])
y = data[['Effort']]'''

#With feature selection using pearson corelation threshold = 0.5
data = pd.read_csv('/Users/tarunchintada/Downloads/desharnais.csv')  # Replace with your actual CSV file
X = data[['Transactions', 'Entities', 'PointsNonAdjust', 'PointsAjust']]
y = data['Effort']

# Normalize the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
model = Sequential()
model.add(Dense(30, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(1))  # Single output neuron for regression

# Compile the model with mean squared error loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')

# Make predictions
predictions = model.predict(X_test)
full_predictions = model.predict(X)

# Print predictions alongside actual values
result = pd.DataFrame({'actual': y.values.flatten(), 'predicted': full_predictions.flatten()})
print(result.to_string(index=False))

mse = mean_squared_error(y_test, predictions)
r2_test = r2_score(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
mmre = mae/np.mean(y_test)

#Return Evalution Metrics
print("---------Evalution Metrics-----------")
print(f'Mean Squared Error: {mse}')

print(f'Mean Absolute Error: {mae}')

print(f'Mean Magnitude of Relative Error : {mmre}')

print(f'Root Mean Squared Error: {rmse}')

print(f'R^2: {r2_test}')




# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
