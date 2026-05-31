import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data set
data = pd.read_csv('data/data.csv')

# Define the behavioral measures for analysis
pca_cols = ['mean_distance', 'mean_angle', 'angle_change', 'mean_hull_area', 'hull_area_change']

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[pca_cols])

# Fit PCA
pca = PCA(n_components=2)
components = pca.fit_transform(scaled_data)

# Add PC scores to the data frame
data['PC1'] = components[:, 0]
data['PC2'] = components[:, 1]

# Get PCA loadings
loadings = pd.DataFrame(pca.components_, columns=pca_cols, index=['PC1', 'PC2'])

# Test whether PC1 is different between the two conditions
human_pc1 = data[data['condition'] == 'human']['PC1']
robot_pc1 = data[data['condition'] == 'robot']['PC1']

# Compute significance levels
u1, p1 = stats.mannwhitneyu(human_pc1, robot_pc1, alternative='two-sided')
d1 = (human_pc1.mean() - robot_pc1.mean()) / np.sqrt(((len(human_pc1)-1)*human_pc1.std()**2 + (len(robot_pc1)-1)*robot_pc1.std()**2) / (len(human_pc1) + len(robot_pc1) - 2))

# Test whether PC2 is different between the two conditions
human_pc2 = data[data['condition'] == 'human']['PC2']
robot_pc2 = data[data['condition'] == 'robot']['PC2']

# Compute significance levels
u2, p2 = stats.mannwhitneyu(human_pc2, robot_pc2, alternative='two-sided')
d2 = (human_pc2.mean() - robot_pc2.mean()) / np.sqrt(((len(human_pc2)-1)*human_pc2.std()**2 + (len(robot_pc2)-1)*robot_pc2.std()**2) / (len(human_pc2) + len(robot_pc2) - 2))

# Save results in a CSV file
data[['trial_id', 'condition', 'PC1', 'PC2']].to_csv('data/pca_results.csv', index=False)


# Create plots
fig, ax = plt.subplots()

# Plot the trials in PC space
human_pca = data[data['condition'] == 'human']
robot_pca = data[data['condition'] == 'robot']
ax.scatter(human_pca['PC1'], human_pca['PC2'], alpha=0.5, color='teal', s=30, label='Human')
ax.scatter(robot_pca['PC1'], robot_pca['PC2'], alpha=0.5, color='darkgoldenrod', s=30, label='Robot')

# Mark the centroid of each condition
ax.scatter(human_pca['PC1'].mean(), human_pca['PC2'].mean(), color='teal', s=150, marker='X', label='Human centroid')
ax.scatter(robot_pca['PC1'].mean(), robot_pca['PC2'].mean(), color='darkgoldenrod', s=150, marker='X', label='Robot centroid')

ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
fig.suptitle(f'Trials in Multimodal Response Space', fontsize=15)
ax.legend(fontsize=8)
ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('plots/pca1.png', dpi=150)

fig, ax = plt.subplots()
# Plot the loadings as barplots
x = np.arange(len(pca_cols))
ax.bar(x - 0.35/2, loadings.loc['PC1'], width=0.35, label='PC1', color='teal')
ax.bar(x + 0.35/2, loadings.loc['PC2'], width=0.35, label='PC2', color='darkgoldenrod')

ax.set_xticks(x)
ax.set_xticklabels(['Distance', 'Angle', 'Angle\nChange', 'Hull\nArea', 'Hull\nChange'], fontsize=9)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Loading')
fig.suptitle('Principal Component Loadings', fontsize=15)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('plots/pca2.png', dpi=150)