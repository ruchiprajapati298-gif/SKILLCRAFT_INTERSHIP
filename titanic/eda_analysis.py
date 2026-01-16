import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set Plot Style
sns.set_theme(style="whitegrid")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_data():
    """Load the Titanic dataset."""
    try:
        df = pd.read_csv('train.csv')
        print("Dataset loaded successfully.")
        return df
    except FileNotFoundError:
        print("Error: train.csv not found.")
        return None

def clean_data(df):
    """Clean the dataset by handling missing values."""
    print(f"\nMissing values before cleaning:\n{df.isnull().sum()}")
    
    # Fill missing Age with median
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Fill missing Embarked with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    
    # Drop Cabin column as it has too many missing values, but creating a feature could be useful later
    # For this simple EDA, we will just create a binary feature 'HasCabin'
    df['HasCabin'] = df['Cabin'].apply(lambda x: 0 if pd.isna(x) else 1)
    df.drop('Cabin', axis=1, inplace=True)
    
    print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")
    return df

def analyze_and_plot(df):
    """Generate and save EDA plots."""
    
    # 1. Survival Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Survived', data=df)
    plt.title('Distribution of Survival (0 = No, 1 = Yes)')
    plt.savefig('plots/survival_distribution.png')
    plt.close()
    
    # 2. Survival by Sex
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Sex', y='Survived', data=df)
    plt.title('Survival Rate by Sex')
    plt.savefig('plots/survival_by_sex.png')
    plt.close()
    
    # 3. Survival by Pclass
    plt.figure(figsize=(6, 4))
    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.title('Survival Rate by Pclass')
    plt.savefig('plots/survival_by_pclass.png')
    plt.close()

    # 4. Age Distribution by Survival
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, element="step")
    plt.title('Age Distribution by Survival Status')
    plt.savefig('plots/age_distribution_by_survival.png')
    plt.close()
    
    # 5. Correlation Matrix (numerical columns only)
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.savefig('plots/correlation_matrix.png')
    plt.close()
    
    print("\nPlots saved to 'plots/' directory.")

def main():
    df = load_data()
    if df is not None:
        print(f"Data Shape: {df.shape}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        df = clean_data(df)
        analyze_and_plot(df)

if __name__ == "__main__":
    main()
