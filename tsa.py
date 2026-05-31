import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Load trajectory files
traj_human = np.load('data/trajectories_human.npy')
traj_robot = np.load('data/trajectories_robot.npy')

# Define the trajectory
trajectory = np.linspace(0, 100, 100)

# Compute means and confidence bands
mean_human = np.nanmean(traj_human, axis=0)
mean_robot = np.nanmean(traj_robot, axis=0)
sem_human = stats.sem(traj_human, axis=0, nan_policy='omit')
sem_robot = stats.sem(traj_robot, axis=0, nan_policy='omit')

p_values = np.zeros(100)
for t in range(100):
    # Extract the angle value at time point t
    human_at_t = traj_human[:, t]
    robot_at_t = traj_robot[:, t]

    # Remove NaN values
    human_at_t = human_at_t[~np.isnan(human_at_t)]
    robot_at_t = robot_at_t[~np.isnan(robot_at_t)]

    # Compute significance levels
    _, p_values[t] = stats.mannwhitneyu(human_at_t, robot_at_t, alternative='two-sided')

# Create a boolean mask for significance
sig = p_values < 0.05

# Get average difference between the two conditions
diff = mean_human.mean() - mean_robot.mean()

# Create the plot
fig, ax = plt.subplots()
fig.suptitle('Time Series Analysis: Body Orientation', fontsize=15)

# Plot mean trajectories with confidence bands
ax.plot(trajectory, mean_human, color='teal', linewidth=2, label='Human newcomer')
ax.plot(trajectory, mean_robot, color='darkgoldenrod', linewidth=2, label='Robot newcomer')
ax.fill_between(trajectory, mean_human - sem_human, mean_human + sem_human, alpha=0.2, color='teal')
ax.fill_between(trajectory, mean_robot - sem_robot, mean_robot + sem_robot, alpha=0.2, color='darkgoldenrod')

# Shade the background if the mean difference is significant
for t in range(100):
    if sig[t]:
        ax.axvspan(trajectory[t], trajectory[min(t+1, 99)], alpha=0.1, color='gray')

sig_patch = mpatches.Patch(color='gray', alpha=0.3, label='p < 0.05')
ax.legend(handles=[ax.lines[0], ax.lines[1], sig_patch], fontsize=9)
ax.set_xlabel('Trial progress (%)')
ax.set_ylabel('Mean orientation angle (°)')
ax.set_title('Mean orientation trajectory')

plt.tight_layout()
plt.savefig('plots/tsa.png', dpi=150)