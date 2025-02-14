import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read all JSON files
files = glob.glob('results/*_metrics_*.json')
print(files)
data = []
filenames = []
for f in files:
    with open(f) as file:
        data.append(json.load(file))
        filenames.append(f)

# Convert to pandas DataFrame
metrics = {'variable': [], 'season': [], 'metric': [], 'value': [], 'simulation_type': [], 'simulation_name': []}
for d, f in zip(data, filenames):
    # Extract simulation name from filename
    sim_name = f.split('_metrics_')[0].split('/')[-1]
    for var in d.keys():
        for season in d[var].keys():
            for metric, value in d[var][season].items():
                metrics['variable'].append(var)
                metrics['season'].append(season)
                metrics['metric'].append(metric)
                metrics['value'].append(value)
                metrics['simulation_name'].append(sim_name)
                if 'nncam_rerun' in f:
                    metrics['simulation_type'].append('nncam_rerun')
                else:
                    metrics['simulation_type'].append('our_simulations')

df = pd.DataFrame(metrics)
print(df)

# Create custom ordering for variables
variable_order = [
    't850', 't500',  # Temperature variables
    'q850', 'q500',  # Humidity variables
    'precip', 't2m'                  # Precipitation
]

# Convert variable column to Categorical with custom ordering
df['variable'] = pd.Categorical(df['variable'], categories=variable_order, ordered=True)
# Modified summary statistics calculation
summary = df[df['simulation_type'] == 'our_simulations'].groupby(['variable', 'season', 'metric']).agg({
    'value': ['mean', 'std', 'min', 'max']
}).round(4)

# Create min/max names separately
def get_min_max_sims(group):
    min_idx = group['value'].idxmin()
    max_idx = group['value'].idxmax()
    return pd.Series({
        ('simulation_name', 'min_sim'): group.loc[min_idx, 'simulation_name'],
        ('simulation_name', 'max_sim'): group.loc[max_idx, 'simulation_name']
    })

min_max_df = df[df['simulation_type'] == 'our_simulations'].groupby(['variable', 'season', 'metric']).apply(get_min_max_sims)

# Combine with summary
summary = pd.concat([summary, min_max_df], axis=1)

# Calculate coefficient of variation (CV)
summary['value', 'cv'] = summary['value', 'std'] / summary['value', 'mean']

# Define variable groups
variable_groups = {
    'Temperature': ['t850', 't500', 't2m'],
    'Humidity': ['q850', 'q500'],
    'Precipitation': ['precip']
}

# Modified plotting code
for metric in ['correlation', 'rmse']:
    fig, axes = plt.subplots(len(variable_groups), 1, 
                            figsize=(15, 5*len(variable_groups)),
                            squeeze=False)
    axes = axes.flatten()
    
    for ax_idx, (group_name, group_vars) in enumerate(variable_groups.items()):
        ax = axes[ax_idx]
        
        # Filter data for current group
        group_data = df[
            (df['metric'] == metric) & 
            (df['variable'].isin(group_vars))
        ]
        
        # Create box plot for group
        sns_plot = sns.boxplot(data=group_data,
                              x='variable', y='value', hue='season',
                              order=group_vars,
                              ax=ax)
        
        # Get the color mapping from the current plot
        season_colors = {text.get_text(): patch.get_facecolor() 
                        for text, patch in zip(ax.legend_.get_texts(), ax.legend_.get_patches())}
        
        # # Remove outlier points

        for line in ax.lines:
            if line.get_linestyle() == 'None':
                line.remove()
        
        # Add stars for nncam_rerun points
        nncam_data = group_data[group_data['simulation_type'] == 'nncam_rerun']
        if not nncam_data.empty:
            seasons = list(df['season'].unique())
            num_seasons = len(seasons)
            
            # Get the actual positions from the boxplot
            num_boxes = len(group_vars) * num_seasons
            positions = np.arange(num_boxes) // num_seasons
            offsets = np.linspace(-0.4, 0.4, num_seasons)
            
            # Create position mapping
            pos_map = {}
            for var_idx, var in enumerate(group_vars):
                for season_idx, season in enumerate(seasons):
                    pos_map[(var, season)] = positions[var_idx * num_seasons + season_idx] + offsets[season_idx]*0.8
            
            # Plot stars using calculated positions
            for idx, row in nncam_data.iterrows():
                x_pos = pos_map[(row['variable'], row['season'])]
                ax.plot(x_pos, row['value'], 
                       marker='*', markersize=10, 
                       color=season_colors[row['season']], 
                       zorder=10)
        
        # Customize subplot
        ax.set_title(f'{group_name} {metric}')
        ax.set_xlabel('')
        ax.set_ylabel(metric.upper())
        
        # Only keep one legend (on the first subplot)
        if ax_idx != 0:
            ax.get_legend().remove()
    
    plt.tight_layout()
    plt.show()

print(summary)