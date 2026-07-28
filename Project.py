# soil_quality_analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_data():
    df = pd.read_csv('soil_quality_dataset.csv')
    print("Data loaded. Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nFirst 5 rows:")
    print(df.head())
    return df

# Show visualizations
def show_visualizations(df):
    # Create a figure with subplots
    plt.figure(figsize=(18, 6))
    
    # 1. Bar Chart - Quality Distribution
    plt.subplot(1, 3, 1)
    quality_counts = df['Quality'].value_counts().sort_index()
    quality_labels = ['Poor', 'Fair', 'Good', 'Excellent']
    colors = ['red', 'orange', 'lightgreen', 'green']
    
    plt.bar(quality_labels, quality_counts.values, color=colors, alpha=0.7)
    plt.title('Soil Quality Distribution - Bar Chart')
    plt.xlabel('Quality Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(quality_counts.values):
        plt.text(i, v + 50, str(v), ha='center', va='bottom')
    
    # 2. Pie Chart
    plt.subplot(1, 3, 2)
    plt.pie(quality_counts.values, labels=quality_labels, colors=colors, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Soil Quality Distribution - Pie Chart')
    
    # 3. Heat Map
    plt.subplot(1, 3, 3)
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                square=True, cbar_kws={"shrink": .8})
    plt.title('Correlation Heat Map')
    plt.tight_layout()
    
    plt.show()
    


# Train model
def train_soil_quality_model(df):
    X = df.drop('Quality', axis=1)
    y = df['Quality']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Poor', 'Fair', 'Good', 'Excellent']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Poor', 'Fair', 'Good', 'Excellent'],
                yticklabels=['Poor', 'Fair', 'Good', 'Excellent'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    


# Crop suggestions based on soil quality
def get_crop_suggestions(quality):
    crop_suggestions = {
        1: {  # Poor Quality
            'Recommended': ['Pearl Millet (Bajra)', 'Sorghum (Jowar)', 'Finger Millet (Ragi)', 
                          'Cowpea', 'Cluster Bean', 'Castor'],
            'Soil Improvement': ['Add organic compost', 'Use green manure', 'Practice crop rotation',
                               'Apply lime/sulfur to adjust pH']
        },
        2: {  # Fair Quality
            'Recommended': ['Maize', 'Barley', 'Oats', 'Sunflower', 'Groundnut', 'Soybean',
                          'Cotton', 'Chickpea'],
            'Soil Improvement': ['Add balanced fertilizers', 'Incorporate crop residue',
                               'Use cover crops', 'Practice conservation tillage']
        },
        3: {  # Good Quality
            'Recommended': ['Wheat', 'Rice', 'Sugarcane', 'Potato', 'Tomato', 'Cabbage',
                          'Cauliflower', 'Onion', 'Garlic'],
            'Soil Improvement': ['Maintain soil organic matter', 'Practice precision farming',
                               'Use drip irrigation', 'Monitor soil nutrients regularly']
        },
        4: {  # Excellent Quality
            'Recommended': ['All high-value crops', 'Vegetables: Bell Pepper, Broccoli, Lettuce',
                          'Fruits: Strawberry, Grapes, Citrus', 'Flowers: Rose, Marigold',
                          'Medicinal plants', 'Organic farming crops'],
            'Soil Improvement': ['Maintain current practices', 'Regular soil testing',
                               'Sustainable farming practices', 'Crop diversification']
        }
    }
    
    quality_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
    suggestions = crop_suggestions[quality]
    
    print(f"\n=== CROP SUGGESTIONS FOR {quality_map[quality].upper()} SOIL QUALITY ===")
    print("\n🌱 RECOMMENDED CROPS:")
    for crop in suggestions['Recommended']:
        print(f"   ✓ {crop}")
    
    print("\n🛠️ SOIL IMPROVEMENT PRACTICES:")
    for practice in suggestions['Soil Improvement']:
        print(f"   • {practice}")

# Predict function
def predict_soil_quality(model):
    print("\n" + "="*50)
    print("SOIL QUALITY PREDICTION")
    print("="*50)
    
    print("\nEnter soil parameters:")
    print("----------------------")
    
    try:
        ph = float(input("pH level (3.5-9.0): "))
        nitrogen = float(input("Nitrogen content mg/kg (5-150): "))
        phosphorus = float(input("Phosphorus content mg/kg (5-100): "))
        potassium = float(input("Potassium content mg/kg (20-400): "))
        organic_matter = float(input("Organic Matter % (0.5-8.0): "))
        moisture = float(input("Moisture content % (5-60): "))
        
        input_data = np.array([[ph, nitrogen, phosphorus, potassium, organic_matter, moisture]])
        prediction = model.predict(input_data)[0]
        
        quality_map = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
        quality_colors = {1: 'red', 2: 'orange', 3: 'lightgreen', 4: 'green'}
        
        print(f"\n" + "="*40)
        print(f"RESULT: Quality {prediction} - {quality_map[prediction]}")
        print("="*40)
        
        # Get crop suggestions
        get_crop_suggestions(prediction)
        
    except ValueError:
        print("Error: Please enter valid numerical values!")
    except Exception as e:
        print(f"Error during prediction: {e}")

# Main execution
def main():
    print("SOIL QUALITY ANALYSIS AND PREDICTION SYSTEM")
    print("="*50)
    
    try:
        # Load data
        df = load_data()
        
        # Show visualizations
        show_visualizations(df)
        
        # Train model
        model, X_test, y_test = train_soil_quality_model(df)
        
        # Evaluate
        evaluate_model(model, X_test, y_test)
        
        # Interactive prediction
        while True:
            predict_soil_quality(model)
            cont = input("\n🔍 Predict again? (y/n): ").lower()
            if cont != 'y':
                print("\nThank you for using Soil Quality Analysis System!")
                break
                
    except FileNotFoundError:
        print("Error: Dataset file 'soil_quality_dataset.csv' not found!")
        print("Please run 'generate_soil_dataset.py' first to generate the dataset.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
