import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_population():
    # File paths
    data_file = 'API_SP.POP.TOTL_DS2_en_csv_v2_174326.csv'
    metadata_file = 'Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_174326.csv'

    print("Loading data...")
    # Load Main Data (skipping first 4 rows of metadata)
    df = pd.read_csv(data_file, skiprows=4)
    
    # Load Metadata
    meta_df = pd.read_csv(metadata_file)

    print("Filtering data...")
    # Identify countries (exclude aggregates where Region is NaN or empty)
    # Check if 'Region' column exists and filter
    if 'Region' in meta_df.columns:
        countries = meta_df[meta_df['Region'].notna() & (meta_df['Region'] != '')]['Country Code'].tolist()
        df_countries = df[df['Country Code'].isin(countries)].copy()
    else:
        print("Warning: 'Region' column not found in metadata. Using all data.")
        df_countries = df.copy()

    # Focus on 2023 data
    if '2023' not in df_countries.columns:
        print("Error: '2023' column not found in dataset.")
        return

    # Extract relevant columns and drop missing values for 2023
    pop_2023 = df_countries[['Country Name', '2023']].dropna()
    
    # Convert population to numeric just in case
    pop_2023['2023'] = pd.to_numeric(pop_2023['2023'])

    # Sort by population
    pop_2023 = pop_2023.sort_values(by='2023', ascending=False)

    print(f"Top 5 most populous countries in 2023:\n{pop_2023.head(5)}")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))

    # 1. Bar Chart: Top 10 Countries
    top_10 = pop_2023.head(10)
    axes[0].bar(top_10['Country Name'], top_10['2023'] / 1e6, color='skyblue')
    axes[0].set_title('Top 10 Most Populous Countries (2023)')
    axes[0].set_ylabel('Population (Millions)')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Histogram: Population Distribution
    # Using log scale for population distribution as it spans many orders of magnitude
    axes[1].hist(pop_2023['2023'] / 1e6, bins=30, color='lightgreen', edgecolor='black')
    axes[1].set_title('Distribution of Country Populations (2023)')
    axes[1].set_xlabel('Population (Millions)')
    axes[1].set_ylabel('Number of Countries')
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    # axes[1].set_yscale('log') # Optional: log scale for y-axis if many small countries

    plt.tight_layout()
    
    output_file = 'population_visualization_2023.png'
    plt.savefig(output_file)
    print(f"Visualization saved to {output_file}")

if __name__ == "__main__":
    visualize_population()
