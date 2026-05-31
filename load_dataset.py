import os
import numpy as np
import pandas as pd
from utils import load_trial, compute_measures, get_csv

# Create output structures
results = []
trajectories = {'human': [], 'robot': []}

# Load data
for condition, folder in [('human', 'human-csv'), ('robot', 'robot-csv')]:
    folder_path = os.path.join('congreg8', folder)

    # Go over all groups
    for group in sorted(os.listdir(folder_path)):
        group_path = os.path.join(folder_path, group)

        # Go over all files in that group
        for name in sorted(os.listdir(group_path)):
            path = os.path.join(group_path, name)

            # Build a more explanatory id for each trial
            trial_id = f'{condition}_{group}_{name}'

            # Load the trials and compute the measures for each trial
            df = load_trial(path)
            df = compute_measures(df, condition)
            results.append(get_csv(df, trial_id, condition))

            # Normalize the orientation angle timeline to 0-100% for time series analysis
            t = np.interp(np.linspace(0, 100, 100), df['time_pct'].values, df['avg_orientation_angle'].values)
            trajectories[condition].append(t)

# Save results in a CSV file
data = pd.DataFrame(results)
data.to_csv('data/data.csv', index=False)

# Save results in a npy file
np.save('data/trajectories_human.npy', np.array(trajectories['human']))
np.save('data/trajectories_robot.npy',  np.array(trajectories['robot']))