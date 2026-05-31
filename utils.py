import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

# Data loading
def load_trial(filepath):

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Split each line into a list of values
    rows = [line.strip().split(',') for line in lines]

    # Get the relevant header rows
    landmark = rows[3]
    posrot = rows[5]
    axis = rows[6]

    # Change the colum names to something more meaningful
    col_names = []
    max_len = max(len(landmark), len(axis))
    for i in range(max_len):
        l = landmark[i].strip() if i < len(landmark) else ''
        p = posrot[i].strip() if i < len(posrot) else ''
        a = axis[i].strip() if i < len(axis) else ''
        col_names.append(f'{l}_{p}_{a}' if l else a)

    # Fix the first two columns at frame and time
    col_names[0] = 'Frame'
    col_names[1] = 'Time'

    # Load data rows after the initial seven header rows
    data_rows = []
    for line in lines[7:]:
        vals = line.strip().split(',')
        # Make sure the columns are the right length
        vals = vals[:len(col_names)]
        while len(vals) < len(col_names):
            vals.append(np.nan)
        data_rows.append(vals)

    df = pd.DataFrame(data_rows, columns=col_names)

    # Drop empty / text rows
    df = df[pd.to_numeric(df['Frame'], errors='coerce').notna()]
    df['Frame'] = df['Frame'].astype(int)
    df['Time']  = df['Time'].astype(float)
    return df


# Get the ids of the participants
def get_all_participants(df):
    # Get all the columns of the hip positions
    hip_cols = []
    for col in df.columns:
        if col.endswith('_Hip_Position_X'):
            hip_cols.append(col)

    # Get the participant's id from each column name
    id = []
    for col in hip_cols:
        prefix = col.replace('_Hip_Position_X', '')
        # Exclude the robot tracker columns
        if not prefix.startswith('extra') and 'Rigid' not in prefix:
            id.append(prefix)
    return id


# Get the XZ position of the newcomer
def get_newcomer_pos(df, condition):

    if condition == 'human':
        # In the human condition, the newcomer is always at p4
        newcomer_id = [p for p in get_all_participants(df)
                             if re.match(r'p4', p)]
        p = newcomer_id[0]
        x = pd.to_numeric(df[f'{p}_Hip_Position_X'], errors='coerce').values
        z = pd.to_numeric(df[f'{p}_Hip_Position_Z'], errors='coerce').values

    else:
        # The robot column always ends in _Position_X
        robot_col_x = None
        for col in df.columns:
            if col.endswith('_Position_X'):
                prefix = col.replace('_Position_X', '')
                # Make sure it's not a person column
                if not prefix.lower().startswith('p'):
                    robot_col_x = col
                    break
        # Get the Z column
        robot_col_z = robot_col_x.replace('_Position_X', '_Position_Z')

        # Convert data from string into numbers
        x = pd.to_numeric(df[robot_col_x], errors='coerce').values
        z = pd.to_numeric(df[robot_col_z], errors='coerce').values
    return x, z


# Get participants ids without the newcomer
def get_fixed_participants(df, condition):

    all_ids = get_all_participants(df)

    # Exclude p4
    if condition == 'human':
        return [p for p in all_ids if not re.match(r'p4', p)][:3]
    else:
        # All named ids are group members
        return all_ids[:3]


# Compute the F-Formation
def compute_fformation(df, group_ids):

    areas = []
    for _, row in df.iterrows():
        # Collect XZ hip positions for all group members in this frame
        points = []
        for p in group_ids:
            x = pd.to_numeric(row.get(f'{p}_Hip_Position_X'), errors='coerce')
            z = pd.to_numeric(row.get(f'{p}_Hip_Position_Z'), errors='coerce')
            if pd.notna(x) and pd.notna(z):
                points.append([x, z])

        # Need at least 3 points to form a hull
        if len(points) >= 3:
            try:
                # Get the smallest polygon containing all participants' hip positions for the f-formation
                hull = ConvexHull(points)
                areas.append(hull.volume)
            except Exception:
                # When all participants stand in one line
                areas.append(np.nan)
        else:
            areas.append(np.nan)
    return np.array(areas)


# Compute the three multimodal measures for one trial
def compute_measures(df, condition):

    newcomer_x, newcomer_z = get_newcomer_pos(df, condition)
    group_ids = get_fixed_participants(df, condition)

    # Convert relevant columns to numbers
    for p in group_ids:
        for ax in ['X', 'Y', 'Z']:
            for part in ['Hip', 'Chest']:
                col = f'{p}_{part}_Position_{ax}'
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

    # Get the average distance to the newcomer
    distances = []
    for p in group_ids:
        dx = df[f'{p}_Hip_Position_X'].values - newcomer_x
        dz = df[f'{p}_Hip_Position_Z'].values - newcomer_z
        distances.append(np.sqrt(dx**2 + dz**2))
    df['avg_distance'] = np.mean(distances, axis=0)

    # Get the average orientation angle toward the newcomer
    angles = []
    for p in group_ids:
        chest_x_col = f'{p}_Chest_Position_X'
        chest_z_col = f'{p}_Chest_Position_Z'
        if chest_x_col not in df.columns:
            continue
        # Get the direction in which the body faces
        facing_x = df[chest_x_col].values - df[f'{p}_Hip_Position_X'].values
        facing_z = df[chest_z_col].values - df[f'{p}_Hip_Position_Z'].values
        # Get the direction from the group member toward the newcomer
        to_new_x = newcomer_x - df[f'{p}_Hip_Position_X'].values
        to_new_z = newcomer_z - df[f'{p}_Hip_Position_Z'].values
        # Get the angle between the two vectors using the dot product
        dot = facing_x * to_new_x + facing_z * to_new_z
        mag1 = np.sqrt(facing_x**2 + facing_z**2)
        mag2 = np.sqrt(to_new_x**2 + to_new_z**2)
        # Handle edge cases
        with np.errstate(invalid='ignore'):
            cos_angle = np.clip(dot / (mag1 * mag2 + 1e-8), -1, 1)
        angles.append(np.degrees(np.arccos(cos_angle)))
    df['avg_orientation_angle'] = np.mean(angles, axis=0)

    # Get the f-formation
    df['hull_area'] = compute_fformation(df, group_ids)

    # Normalize the time to percentages for time series analysis
    t_min, t_max = df['Time'].min(), df['Time'].max()
    df['time_pct'] = (df['Time'] - t_min) / (t_max - t_min) * 100
    return df


# Create a CSV file of all the measures
def get_csv(df, trial_id, condition):

    angle_series = df['avg_orientation_angle'].dropna()
    hull_series  = df['hull_area'].dropna()

    return {
        'trial_id':         trial_id,
        'condition':        condition,
        'n_frames':         len(df),
        'duration_s':       len(df) / 120,
        'mean_distance':    df['avg_distance'].mean(),
        'min_distance':     df['avg_distance'].min(),
        'mean_angle':       angle_series.mean(),
        'final_angle':      angle_series.iloc[-1] if len(angle_series) > 0 else np.nan,
        'angle_change':     (angle_series.iloc[0] - angle_series.iloc[-1])
                            if len(angle_series) > 0 else np.nan,
        'mean_hull_area':   hull_series.mean(),
        'hull_area_change': (hull_series.iloc[-1] - hull_series.iloc[0])
                            if len(hull_series) > 0 else np.nan,
    }