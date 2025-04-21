import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set Seaborn style
sns.set_theme(style="whitegrid")


def plot_column(df, column, kind='bar', title=None, ylabel=None, palette='viridis'):
    """
    Beautifully plots a given column from the DataFrame against the 'model' column.

    Parameters:
    - df: DataFrame containing the data
    - column: Name of the column to plot
    - kind: Type of plot ('bar', 'line')
    - title: Custom title (optional)
    - ylabel: Custom Y-axis label (optional)
    - palette: Color palette name (default 'viridis')
    """
    plt.figure(figsize=(14, 7))

    # Sorting for better visuals
    df_sorted = df.sort_values(by=column, ascending=False)

    # Bar Plot
    if kind == 'bar':
        ax = sns.barplot(
            x='model', y=column, data=df_sorted,
            palette=palette, edgecolor='black'
        )
        # Add data labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', fontsize=10, label_type='edge', padding=3)

    # Line Plot
    elif kind == 'line':
        ax = sns.lineplot(
            x='model', y=column, data=df_sorted,
            marker='o', markersize=8, linewidth=2.5, palette=palette
        )
        # Add value annotations
        for i, row in df_sorted.iterrows():
            plt.text(i, row[column], f'{row[column]:.2f}', ha='center', va='bottom', fontsize=10)

    else:
        raise ValueError("Unsupported plot kind. Use 'bar' or 'line'.")

    # General Aesthetic
    plt.title(title if title else f"{column} by Model", fontsize=16, weight='bold')
    plt.ylabel(ylabel if ylabel else column, fontsize=12)
    plt.xlabel("Evaluation Strategy", fontsize=12)
    plt.xticks(rotation=25, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
