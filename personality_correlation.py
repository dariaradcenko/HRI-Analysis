import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


# Load the data
questionnaire = pd.ExcelFile('data/questionnaire.xlsx')
personality = pd.read_excel(questionnaire, sheet_name='Personality', header=None)

# Define column names based on the descriptions from the data set
personality.columns = [
    'participant_id',
    'extraverted',
    'critical',
    'dependable',
    'anxious',
    'open',
    'reserved',
    'sympathetic',
    'disorganized',
    'calm',
    'conventional',
]

# Drop the NaN rows
personality = personality.dropna(subset=['participant_id']).reset_index(drop=True)

# Compute the personality score according to the TIPI scheme
def reverse(x, scale=7):
    return (scale + 1) - x

personality['O'] = (personality['open'] + reverse(personality['conventional'])) / 2
personality['C'] = (personality['dependable'] + reverse(personality['disorganized'])) / 2
personality['E'] = (personality['extraverted'] + reverse(personality['reserved'])) / 2
personality['A'] = (personality['sympathetic'] + reverse(personality['critical'])) / 2
personality['N'] = (personality['anxious'] + reverse(personality['calm'])) / 2

# Get the participant based on the id
def get_participant(id):
    # Convert letters into numbers
    id = str(id).strip()
    letter = id[0].lower()
    group_num = ord(letter) - ord('a') + 1
    number = int(id[1:])
    return group_num, number

personality[['group_num', 'person_num']] = personality['participant_id'].apply(lambda x: pd.Series(get_participant(x)))

# Load the data set
data = pd.read_csv('data/data.csv')

# Get group number from trial_id
def get_group(trial_id):
    m = re.search(r'Group(\d+)', trial_id)
    return int(m.group(1)) if m else None

data['group_num'] = data['trial_id'].apply(get_group)

# Compute average behavior per group
group_behavior = data.groupby(['group_num', 'condition'])[['mean_distance', 'min_distance', 'mean_angle', 'angle_change', 'mean_hull_area', 'hull_area_change']].mean().reset_index()

# Assign the average behavioral measures of the group to each participant
combined = personality.merge(group_behavior, on='group_num', how='left')

# Define labels
big_five = ['O', 'C', 'E', 'A', 'N']
big_five_labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
behavior = ['mean_distance', 'mean_angle', 'mean_hull_area', 'hull_area_change']
behavior_labels = ['Mean Distance', 'Mean Angle', 'Hull Area', 'Hull Change']



# Filter for human condition
human_combined = combined[combined['condition'] == 'human'].dropna(subset=['O', 'C', 'E', 'A', 'N', 'mean_distance', 'mean_angle','mean_hull_area', 'hull_area_change'])

# Get the Spearman correlations between the personality traits and the behavioral measures
human_corr_results = []
for trait, tlabel in zip(big_five, big_five_labels):
    for behav, blabel in zip(behavior, behavior_labels):
        r, p = stats.spearmanr(human_combined[trait], human_combined[behav])
        human_corr_results.append({
            'trait':    tlabel,
            'behavior': blabel,
            'r':        r,
            'p':        p
        })

# Save the correlation results
human_corr_df = pd.DataFrame(human_corr_results)
human_corr_df.to_csv('data/human_personality_correlations.csv', index=False)


# Create plots for the human condition
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Personality and Multimodal Behavior: Human Condition', fontsize=15)

# Create heatmap
r_matrix = human_corr_df.pivot(index='trait', columns='behavior', values='r').reindex(big_five_labels)
p_matrix = human_corr_df.pivot(index='trait', columns='behavior', values='p').reindex(big_five_labels)

im = axes[0].imshow(r_matrix.values, cmap='BrBG', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(behavior_labels)))
axes[0].set_yticks(range(len(big_five_labels)))
axes[0].set_xticklabels(behavior_labels, rotation=20, ha='right', fontsize=9)
axes[0].set_yticklabels(big_five_labels, fontsize=9)
axes[0].set_title('Personality vs Behavior (Spearman r (p < 0.05))')
plt.colorbar(im, ax=axes[0], label='Correlation r')

# Annotate cells
for i, j in zip(range(len(big_five_labels)), range(len(behavior_labels))):
    r_val = r_matrix.values[i, j]
    axes[0].text(j, i, f'{r_val:+.2f}', ha='center', va='center', fontsize=9, color='black')

# Create the scatter plot with the strongest correlation
human_strongest = human_corr_df.loc[human_corr_df['r'].abs().idxmax()]
human_traits = big_five[big_five_labels.index(human_strongest['trait'])]
human_behaviors = behavior[behavior_labels.index(human_strongest['behavior'])]

x = human_combined[human_traits].astype(float)
y = human_combined[human_behaviors].astype(float)
m, b   = np.polyfit(human_combined[human_traits].astype(float), human_combined[human_behaviors].astype(float), 1)
x_line = np.linspace(x.min(), x.max(), 100)

axes[1].scatter(x, y, alpha=0.6, color='teal', s=60)
axes[1].plot(x_line, m * x_line + b, color='darkgoldenrod', linewidth=2)
axes[1].set_xlabel(human_strongest['trait'], fontsize=10)
axes[1].set_ylabel(human_strongest['behavior'], fontsize=10)
axes[1].set_title(f"Strongest correlation (r={human_strongest['r']:+.3f}, p={human_strongest['p']:.4f})")

plt.tight_layout()
plt.savefig('plots/human_personality_correlations.png', dpi=150)



# Filter behavior for the robot condition
robot_behavior = group_behavior[group_behavior['condition'] == 'robot']
robot_combined = personality.merge(robot_behavior, on='group_num', how='inner').dropna(subset=['O', 'C', 'E', 'A', 'N', 'mean_distance', 'mean_angle','mean_hull_area', 'hull_area_change'])

# Get the Spearman correlations between the personality traits and the behavioral measures
robot_corr_results = []
for trait, tlabel in zip(big_five, big_five_labels):
    for behav, blabel in zip(behavior, behavior_labels):
        r, p = stats.spearmanr(robot_combined[trait], robot_combined[behav])
        robot_corr_results.append({
            'trait':    tlabel,
            'behavior': blabel,
            'r':        r,
            'p':        p
        })

# Save the correlation results
robot_corr_df = pd.DataFrame(robot_corr_results)
robot_corr_df.to_csv('data/robot_personality_correlations.csv', index=False)

# Create plots
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Personality and Multimodal Behavior: Robot Condition', fontsize=15)

# Create heatmap
r_matrix = robot_corr_df.pivot(index='trait', columns='behavior', values='r').reindex(big_five_labels)
p_matrix = robot_corr_df.pivot(index='trait', columns='behavior', values='p').reindex(big_five_labels)

im = axes[0].imshow(r_matrix.values, cmap='BrBG', vmin=-1, vmax=1, aspect='auto')
axes[0].set_xticks(range(len(behavior_labels)))
axes[0].set_yticks(range(len(big_five_labels)))
axes[0].set_xticklabels(behavior_labels, rotation=20, ha='right', fontsize=9)
axes[0].set_yticklabels(big_five_labels, fontsize=9)
axes[0].set_title('Personality vs Behavior (Spearman r (p < 0.05))')
plt.colorbar(im, ax=axes[0], label='Correlation r')

# Annotate cells
for i in range(len(big_five_labels)):
    for j in range(len(behavior_labels)):
        r_val    = r_matrix.values[i, j]
        axes[0].text(j, i, f'{r_val:+.2f}',ha='center', va='center', fontsize=9, color='black')

# Create the scatter plot with the strongest correlation
robot_strongest = robot_corr_df.loc[robot_corr_df['r'].abs().idxmax()]
robot_traits = big_five[big_five_labels.index(robot_strongest['trait'])]
robot_behaviors = behavior[behavior_labels.index(robot_strongest['behavior'])]

x = robot_combined[robot_traits].astype(float)
y = robot_combined[robot_behaviors].astype(float)
m, b   = np.polyfit(robot_combined[robot_traits].astype(float), robot_combined[robot_behaviors].astype(float), 1)
x_line = np.linspace(x.min(), x.max(), 100)

axes[1].scatter(x, y, alpha=0.6, color='teal', s=60)
axes[1].plot(x_line, m * x_line + b, color='darkgoldenrod', linewidth=2)
axes[1].set_xlabel(robot_strongest['trait'], fontsize=10)
axes[1].set_ylabel(robot_strongest['behavior'], fontsize=10)
axes[1].set_title(f"Strongest correlation (r={robot_strongest['r']:+.3f}, p={robot_strongest['p']:.4f})")

plt.tight_layout()
plt.savefig('plots/robot_personality_correlations.png', dpi=150)