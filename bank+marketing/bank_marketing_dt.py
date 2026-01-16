import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Set Plot Style
sns.set_theme(style="whitegrid")

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def load_and_preprocess_data():
    """Load and preprocess the Bank Marketing data."""
    print("Loading data...")
    try:
        # The dataset uses ';' as delimiter and has 'unknown' values
        df = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return None, None
    
    print(f"Data Loaded. Shape: {df.shape}")
    
    # Preprocessing
    # Convert 'y' (target) to binary (yes=1, no=0)
    df['y'] = df['y'].apply(lambda x: 1 if x == 'yes' else 0)
    
    # Handle categorical variables (One-Hot Encoding)
    # We will simply get dummies for all object columns except 'y' (already converted)
    categorical_cols = df.select_dtypes(include=['object']).columns
    print(f"Encoding categorical columns: {list(categorical_cols)}")
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']
    
    return X, y

def train_and_evaluate(X, y):
    """Train Decision Tree and evaluate performance."""
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Initialize and Train Model
    clf = DecisionTreeClassifier(random_state=42, max_depth=5) # Limiting depth for visibility in plot
    print("Training Decision Tree Classifier...")
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_test)
    
    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('plots/confusion_matrix.png')
    plt.close()
    
    # Plot Decision Tree
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, fontsize=10)
    plt.title('Decision Tree Visualization (Max Depth = 5)')
    plt.savefig('plots/decision_tree.png')
    plt.close()
    
    print("Plots saved to 'plots/' directory.")

def main():
    X, y = load_and_preprocess_data()
    if X is not None:
        train_and_evaluate(X, y)

if __name__ == "__main__":
    main()
