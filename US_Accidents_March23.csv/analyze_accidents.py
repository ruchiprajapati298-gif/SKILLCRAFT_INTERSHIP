import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium import plugins
import os
import random

# Set Plot Style
sns.set_theme(style="whitegrid")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

DATA_FILE = 'US_Accidents_March23.csv'
SAMPLE_SIZE = 200000 

def load_data_sample():
    """Load a random sample of the dataset to save memory."""
    print(f"Counting lines in {DATA_FILE}...")
    # Get total rows (approximate or counting directly can be slow, but for 3GB maybe acceptable. 
    # Alternatively, we can skip rows randomly)
    
    # Efficient Sampling Strategy:
    # 1. Read header
    # 2. Skip rows randomly
    
    # Note: For simplicity and speed in this environment, we will use pandas read_csv with skiprows and lambda
    # But estimating line count first is better.
    # Let's assume ~7.7 million rows based on file size/known dataset size. 
    # Logic: Read 200k rows.
    
    print(f"Loading sample of {SAMPLE_SIZE} rows...")
    try:
        # Probabilistic sampling: if ~7.7M rows, we want ~200k. p ~ 0.026
        # A safer way without knowing exact length is just to read the first N rows or use skiprows
        # But random sampling is better for distribution.
        
        # We will read the file in chunks and sample from chunks to be memory efficient and truly random
        df_list = []
        chunksize = 1000000
        for chunk in pd.read_csv(DATA_FILE, chunksize=chunksize):
            df_list.append(chunk.sample(frac=SAMPLE_SIZE/7700000 * 1.5, random_state=42)) # Over-sample slightly to ensure enough data
            # Adjust denom based on approx size or break if we have enough
            if sum(len(c) for c in df_list) > SAMPLE_SIZE:
                break
                
        df = pd.concat(df_list)
        df = df.sample(n=min(len(df), SAMPLE_SIZE), random_state=42) # Trim to exact size
        
        print(f"Sample Loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Preprocess time and other columns."""
    print("Preprocessing...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df = df.dropna(subset=['Start_Time', 'Start_Lat', 'Start_Lng'])
    return df

def analyze_and_plot(df):
    """Generate visualizations."""
    
    # 1. Accidents by Time of Day
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Hour', data=df, palette='viridis')
    plt.title('Accidents by Time of Day')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Accidents')
    plt.savefig('plots/accidents_by_hour.png')
    plt.close()
    
    # 2. Top 10 Weather Conditions
    plt.figure(figsize=(10, 6))
    top_weather = df['Weather_Condition'].value_counts().head(10)
    sns.barplot(y=top_weather.index, x=top_weather.values, palette='coolwarm')
    plt.title('Top 10 Weather Conditions for Accidents')
    plt.xlabel('Number of Accidents')
    plt.savefig('plots/accidents_by_weather.png')
    plt.close()
    
    # 3. Road Conditions Feature Impact
    road_features = ['Traffic_Signal', 'Junction', 'Crossing', 'Station', 'Stop']
    feature_counts = df[road_features].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_counts.index, y=feature_counts.values, palette='magma')
    plt.title('Accidents near Road Features')
    plt.xlabel('Feature')
    plt.ylabel('Count')
    plt.savefig('plots/accidents_road_features.png')
    plt.close()

    # 4. Folium Heatmap (Hotspots)
    print("Generating Heatmap...")
    # Use a smaller subset for the map to keep it responsive
    map_data = df[['Start_Lat', 'Start_Lng']].sample(n=min(len(df), 10000), random_state=42)
    
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4) # Center of US
    heat_data = [[row['Start_Lat'], row['Start_Lng']] for index, row in map_data.iterrows()]
    plugins.HeatMap(heat_data).add_to(m)
    
    m.save('accident_hotspots.html')
    print("Analysis Complete. Outputs saved.")

def main():
    df = load_data_sample()
    if df is not None:
        df = preprocess_data(df)
        analyze_and_plot(df)

if __name__ == "__main__":
    main()
