#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:07:40 2024

@author: tarunchintada
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
data = pd.read_csv(r"/Users/tarunchintada/Documents/NITW Research/cocomo81.csv")  # Update with your dataset path
X = data.drop(columns=['actual'])
y = data['actual']

# Set a random seed for reproducibility
np.random.seed(42)

# Function to calculate MMRE
def calculate_mmre(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))

# Distance function
def dis(f1, f2, f_type='numerical'):
    if f_type == 'numerical' or f_type == 'ordinal':
        return f1 - f2
    elif f_type == 'nominal':
        return 0 if f1 == f2 else 1

# Similarity function
def similarity(p1, p2, weights, delta=0.0001):
    n = len(p1)  # Number of features
    dist = 0
    for i in range(n):
        dist += weights[i] * abs(dis(p1[i], p2[i]))  # Ensure non-negative distances
    dist = np.sqrt(dist + delta)   
    return 1 / dist

# Calculate effort
def calculate_effort(new_project, train_features, train_target, weights, k=3):
    similarities = np.array([similarity(new_project, train_features[i], weights) for i in range(len(train_features))])
    sorted_indices = np.argsort(-similarities)[:k]
    sim_sum = np.sum(similarities[sorted_indices])
    effort_estimate = 0
    for index in sorted_indices:
        effort_estimate += (similarities[index] / sim_sum) * train_target.iloc[index]
    return effort_estimate

# Evaluate fireflies
def evaluate_firefly(firefly, X_train, X_test, y_train, y_test):
    test_estimates = []
    for i in range(X_test.shape[0]):
        new_project = X_test[i]
        estimate = calculate_effort(new_project, X_train, y_train, firefly, k=3)
        test_estimates.append(estimate)
    test_estimates = np.array(test_estimates)
    mmre = calculate_mmre(y_test.values, test_estimates)
    return mmre

# Firefly algorithm parameters
num_fireflies = 10
max_iter = 100
alpha = 0.2
beta = 1.0
gamma = 1.0

# Initialize fireflies
fireflies = [np.random.rand(X.shape[1]) for _ in range(num_fireflies)]
for firefly in fireflies:
    firefly /= np.sum(firefly)  # Normalize weights

# Perform the evaluation over multiple runs
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Store fitness values
fitness_values = [evaluate_firefly(firefly, X_train, X_test, y_train, y_test) for firefly in fireflies]

# Iterate through the algorithm
for iteration in range(max_iter):
    for i in range(num_fireflies):
        for j in range(num_fireflies):
            if fitness_values[j] < fitness_values[i]:
                r = np.linalg.norm(fireflies[i] - fireflies[j])
                beta_t = beta * np.exp(-gamma * r ** 2)
                fireflies[i] = fireflies[i] + beta_t * (fireflies[j] - fireflies[i]) + alpha * (np.random.rand(X.shape[1]) - 0.5)
                fireflies[i] = np.clip(fireflies[i], 0, 1)  # Ensure non-negative weights
                fireflies[i] /= np.sum(fireflies[i])  # Normalize weights
                fitness_values[i] = evaluate_firefly(fireflies[i], X_train, X_test, y_train, y_test)

# Select the best firefly
best_firefly = fireflies[np.argmin(fitness_values)]

# Print optimized weights
print("Optimized Feature Weights:")
print(best_firefly)
print(sum(best_firefly))

# Final evaluation
test_estimates = []
for i in range(X_test.shape[0]):
    new_project = X_test[i]
    estimate = calculate_effort(new_project, X_train, y_train, firefly, k=3)
    test_estimates.append(estimate)
test_estimates = np.array(test_estimates)

optim_test_estimates = []
for i in range(X_test.shape[0]):
    new_project = X_test[i]
    estimate = calculate_effort(new_project, X_train, y_train, best_firefly, k=3)
    optim_test_estimates.append(estimate)
optim_test_estimates = np.array(optim_test_estimates)

mmre = calculate_mmre(y_test.values, test_estimates)
mae = mean_absolute_error(y_test, test_estimates)
mse = mean_squared_error(y_test, test_estimates)
rmse = np.sqrt(mse)

opti_mmre = calculate_mmre(y_test.values, optim_test_estimates)
opti_mae = mean_absolute_error(y_test, optim_test_estimates)
opti_mse = mean_squared_error(y_test, optim_test_estimates)
opti_rmse = np.sqrt(opti_mse)

print("Before Optimization\n")
print("Final MMRE: {:.4f}".format(mmre))
print("Final MAE: {:.4f}".format(mae))
print("Final MSE: {:.4f}".format(mse))
print("Final RMSE: {:.4f}".format(rmse))

print("\nAfter Optimization\n")
print("Final MMRE: {:.4f}".format(opti_mmre))
print("Final MAE: {:.4f}".format(opti_mae))
print("Final MSE: {:.4f}".format(opti_mse))
print("Final RMSE: {:.4f}".format(opti_rmse))

# Collect metrics
metrics = ['MMRE', 'MAE', 'MSE', 'RMSE']
before_optimization = [mmre, mae, mse, rmse]
after_optimization = [opti_mmre, opti_mae, opti_mse, opti_rmse]

# Create a DataFrame for the results
results_df = pd.DataFrame({
    'Metric': metrics,
    'Before Optimization': before_optimization,
    'After Optimization': after_optimization
})

# Plot the bar graph
# Plot the bar graph
fig, ax = plt.subplots(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(metrics))

bar1 = ax.bar(index, results_df['Before Optimization'], bar_width, label='Before Optimization')
bar2 = ax.bar(index + bar_width, results_df['After Optimization'], bar_width, label='After Optimization')

ax.set_xlabel('Metric')
ax.set_ylabel('Value')
ax.set_title('Metrics Before and After Optimization using Firefly Algorithm')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics)
ax.legend()

# Add value labels on bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{:.2e}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bar1)
autolabel(bar2)

# Adjust y-axis scale to scientific notation
ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
ax.set_ylim(1e-3, 1e10) 

plt.tight_layout()
plt.show()

