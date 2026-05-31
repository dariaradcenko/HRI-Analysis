import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the data
data  = pd.read_csv('data/data.csv')

# Filter the two conditions
human = data[data['condition'] == 'human']
robot = data[data['condition'] == 'robot']

# Define the behavioral measures
measures = {
    'Mean Distance (m)': 'mean_distance',
    'Min Distance (m)': 'min_distance',
    'Mean Orientation Angle (°)': 'mean_angle',
    'Angle Change (°)': 'angle_change',
}

results = []
for label, col in measures.items():
    h = human[col].dropna()
    r = robot[col].dropna()

    # Compute significance levels
    u, p = stats.mannwhitneyu(h, r, alternative='two-sided')
    d = (h.mean() - r.mean()) / np.sqrt(((len(h)-1)*h.std()**2 + (len(r)-1)*r.std()**2) / (len(h) + len(r) - 2))

    # Store the measures
    results.append({
        'measure':     label,
        'human_mean':  h.mean(),
        'human_std':   h.std(),
        'robot_mean':  r.mean(),
        'robot_std':   r.std(),
        'U':           u,
        'p':           p,
        'cohens_d':    d,
    })

# Save results in a CSV file
pd.DataFrame(results).to_csv('data/stat_results.csv', index=False)

# Get the correlations between all pairs of modalities
for cond in ['human', 'robot', 'all']:
    sub = data if cond == 'all' else data[data['condition'] == cond]
    sub = sub.dropna(subset=['mean_distance', 'mean_angle', 'mean_hull_area'])

    # Compute Spearman correlation between each pair of modalities
    r1, p1 = stats.spearmanr(sub['mean_distance'], sub['mean_angle'])
    r2, p2 = stats.spearmanr(sub['mean_distance'], sub['mean_hull_area'])
    r3, p3 = stats.spearmanr(sub['mean_angle'],    sub['mean_hull_area'])


distance_measures = {
    'Mean Distance (m)': 'mean_distance',
    'Min Distance (m)':  'min_distance',
}
angle_measures = {
    'Mean Orientation Angle (°)': 'mean_angle',
    'Angle Change (°)':           'angle_change',
}

# Create distance plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Distance to Newcomer: Human vs Robot', fontsize=15)

for ax, (label, col) in zip(axes, distance_measures.items()):
    h = human[col].dropna()
    r = robot[col].dropna()
    _, p = stats.mannwhitneyu(h, r, alternative='two-sided')
    bp   = ax.boxplot([h, r], labels=['Human', 'Robot'], patch_artist=True, medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('teal')
    bp['boxes'][1].set_facecolor('darkgoldenrod')
    ax.set_title(f'{label} (p={p:.4f})', fontsize=9)
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig('plots/stats_distance.png', dpi=150)

# Create angle plots
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle('Body Orientation: Human vs Robot', fontsize=15)

for ax, (label, col) in zip(axes, angle_measures.items()):
    h = human[col].dropna()
    r = robot[col].dropna()
    _, p = stats.mannwhitneyu(h, r, alternative='two-sided')
    bp   = ax.boxplot([h, r], labels=['Human', 'Robot'], patch_artist=True, medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('teal')
    bp['boxes'][1].set_facecolor('darkgoldenrod')
    ax.set_title(f'{label} (p={p:.4f})', fontsize=9)
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig('plots/stats_angle.png', dpi=150)