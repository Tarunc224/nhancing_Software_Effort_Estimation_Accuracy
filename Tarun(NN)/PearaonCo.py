import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
# Replace 'your_data.csv' with the path to your CSV file
df = pd.read_csv('/Users/tarunchintada/Documents/NITW Research/02.desharnais.csv')

# Select only numerical columns
df_numeric = df.select_dtypes(include=[float, int])

# Compute the correlation matrix
corr_matrix = df_numeric.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Find and print attribute pairs with correlation greater than 0.5
threshold = 0.5
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            high_corr_pairs.append(pair)

print("Attribute pairs with correlation greater than 0.5:")
for pair in high_corr_pairs:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")
