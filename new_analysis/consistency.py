import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
nncam_df = df[df['simulation_type'] == 'nncam_rerun']
print(nncam_df)

# Combine with summary
# summary = pd.concat([summary, min_max_df], axis=1)

# Calculate coefficient of variation (CV)
summary['value', 'cv'] = summary['value', 'std'] / summary['value', 'mean']

# Perform ANOVA and add to summary
def perform_anova(group):
    season_groups = [group[group['season'] == season]['value'] for season in group['season'].unique()]
    f_stat, p_val = stats.f_oneway(*season_groups)
    return pd.Series({'f_stat': f_stat, 'p_val': p_val})

# Calculate ANOVA for each variable/metric combination
anova_results = df[df['simulation_type'] == 'our_simulations'].groupby(['variable', 'metric']).apply(perform_anova)

# Create a dictionary to store the p-values for easy lookup
anova_dict = {}
for (var, metric), row in anova_results.iterrows():
    anova_dict[(var, metric)] = row['p_val']

# Add ANOVA results to summary as a new column
summary[('value', 'anova_p')] = [anova_dict[(var, met)] for var, season, met in summary.index]

# Round most columns to 4 decimal places, but use scientific notation with 8 digits for p-values
summary = summary.round({('value', 'mean'): 4, 
                        ('value', 'std'): 4, 
                        ('value', 'min'): 4, 
                        ('value', 'max'): 4,
                        ('value', 'cv'): 4})
summary[('value', 'anova_p')] = summary[('value', 'anova_p')].apply(lambda x: f"{x:.8e}")

# Define variable groups with units
variable_groups = {
    'Temperature': {
        'vars': ['t850', 't500', 't2m'],
        'units': {'correlation': '', 'rmse': 'K'}
    },
    'Humidity': {
        'vars': ['q850', 'q500'],
        'units': {'correlation': '', 'rmse': 'kg/kg'}
    },
    'Precipitation': {
        'vars': ['precip'],
        'units': {'correlation': '', 'rmse': 'mm/day'}
    }
}

plots_label = ['a', 'b', 'c', 'd', 'e', 'f']
# Modified plotting code
for metric in ['correlation', 'rmse']:
    # Create figure for seasonal plots
    fig, axes = plt.subplots(1, len(variable_groups),
                            figsize=(6*len(variable_groups), 5),
                            squeeze=False)
    axes = axes.flatten()
    
    # First plot the variable groups as before
    for ax_idx, (group_name, group_info) in enumerate(variable_groups.items()):
        ax = axes[ax_idx]
        group_vars = group_info['vars']
        unit = group_info['units'][metric]
        
        # Filter data for current group
        group_data = df[
            (df['metric'] == metric) & 
            (df['variable'].isin(group_vars))
        ]
        
        # Create box plot for group
        sns_plot = sns.boxplot(data=group_data[group_data['simulation_type'] == 'our_simulations'],
                              x='variable', y='value', hue='season',
                              order=group_vars,
                              ax=ax)
        
        # Get the color mapping from the current plot
        season_colors = {text.get_text(): patch.get_facecolor() 
                        for text, patch in zip(ax.legend_.get_texts(), ax.legend_.get_patches())}
        
        # Remove outlier points
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
        if metric == 'correlation':
            ax.set_title(f'({plots_label[ax_idx]}) {group_name} {metric}')
        else:
            ax.set_title(f'({plots_label[ax_idx+3]}) {group_name} {metric}')
        ax.set_xlabel('')
        # Add unit to y-label if it exists
        ylabel = metric.upper()
        if unit:
            ylabel += f' ({unit})'
        ax.set_ylabel(ylabel)
        
        # Only keep season legend on first subplot
        if ax_idx != 0:
            ax.get_legend().remove()
        else:
            season_legend = ax.legend(title='Season')
            # Get legend texts and capitalize them
            for text in season_legend.get_texts():
                if text.get_text() == 'annual':
                    text.set_text('Annual')
                else:
                    text.set_text(text.get_text().upper())
            ax.add_artist(season_legend)
            
        # Add Wang2022 legend to every subplot
        wang_line = ax.plot([], [], marker='*', markersize=10, color='gray', linestyle='None', label='Wang2022')[0]
        if metric == 'rmse' and (group_name == 'Precipitation' or group_name == 'Temperature'):
            wang_legend = ax.legend(handles=[wang_line], loc='lower right')
        else:
            wang_legend = ax.legend(handles=[wang_line], loc='upper right')
        ax.add_artist(wang_legend)

    plt.tight_layout()
    plt.show()

    # Create figure with two subplots - one for humidity variables and one for others
    humidity_vars = ['q500', 'q850']
    other_vars = [v for v in variable_order if v not in humidity_vars]
    
    # plt.rcParams.update({'font.size': 14})  # Increase font size
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    annual_data = df[
        (df['metric'] == metric) &
        (df['season'] == 'annual')
    ]
    
    # Plot for non-humidity variables
    non_humidity_data = annual_data[annual_data['variable'].isin(other_vars)]
    sns.boxplot(data=non_humidity_data[non_humidity_data['simulation_type'] == 'our_simulations'],
                x='variable', y='value',
                order=other_vars,
                ax=ax1)
    
    # Add stars for nncam_rerun points
    nncam_annual = non_humidity_data[non_humidity_data['simulation_type'] == 'nncam_rerun']
    if not nncam_annual.empty:
        for idx, row in nncam_annual.iterrows():
            var_idx = other_vars.index(row['variable'])
            ax1.plot(var_idx, row['value'],
                    marker='*', markersize=10,
                    color='gray',
                    zorder=10)
    
    # Customize first subplot
    ax1.set_title(f'Annual {metric} (Temperature & Precipitation)', fontsize=16)
    ax1.set_xlabel('')
    ylabel = metric.upper()
    if unit:
        if any(var in other_vars for var in ['t500', 't850', 't2m']):
            ylabel += ' (K | mm/day)'
        elif 'precip' in other_vars:
            ylabel += ' (mm/day)'
    ax1.set_ylabel(ylabel, fontsize=14)
    # ax1.set_yscale('log')
    ax1.tick_params(axis='both', labelsize=12)
    
    # Add Wang2022 legend to first subplot
    wang_line = ax1.plot([], [], marker='*', markersize=10, color='gray', linestyle='None', label='Wang2022')[0]
    wang_legend = ax1.legend(handles=[wang_line], loc='lower right', fontsize=12)
    ax1.add_artist(wang_legend)
    
    # Plot for humidity variables
    humidity_data = annual_data[annual_data['variable'].isin(humidity_vars)]
    sns.boxplot(data=humidity_data[humidity_data['simulation_type'] == 'our_simulations'],
                x='variable', y='value',
                order=humidity_vars,
                ax=ax2)
    
    # Add stars for nncam_rerun points
    nncam_humidity = humidity_data[humidity_data['simulation_type'] == 'nncam_rerun']
    if not nncam_humidity.empty:
        for idx, row in nncam_humidity.iterrows():
            var_idx = humidity_vars.index(row['variable'])
            ax2.plot(var_idx, row['value'],
                    marker='*', markersize=10,
                    color='gray',
                    zorder=10)
    
    # Customize second subplot
    ax2.set_title(f'Annual {metric} (Humidity)', fontsize=16)
    ax2.set_xlabel('')
    ylabel = metric.upper()
    if unit:
        ylabel += ' (g/kg)'
    ax2.set_ylabel(ylabel, fontsize=14)
    # ax2.set_yscale('log')
    ax2.tick_params(axis='both', labelsize=12)
    
    # Add Wang2022 legend to second subplot
    wang_line = ax2.plot([], [], marker='*', markersize=10, color='gray', linestyle='None', label='Wang2022')[0]
    wang_legend = ax2.legend(handles=[wang_line], loc='lower right', fontsize=12)
    ax2.add_artist(wang_legend)
    
    plt.tight_layout()
    plt.savefig(f'annual_{metric}_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.show()

print(summary)