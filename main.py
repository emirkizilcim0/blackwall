import warnings
warnings.filterwarnings('ignore')

from src.data_loader import NSLKDDLoader
from src.preprocessor import CyberPreprocessor
from src.models import BlackWallModels
from src.evaluator import CyberEvaluator
import pandas as pd
import numpy as np

def main():
    print("""
    ╔═══════════════════════════════════════════════╗
    ║                 BLACKWALL                     ║
    ║       Cyberpunk ML Intrusion Detection        ║ 
    ║                                               ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    try:
        # 🎯 Step 1: Load Data
        print("📥 Phase 1: Loading Cyber Threat Data...")
        loader = NSLKDDLoader()
        train_df, test_df = loader.load_data(
            "data/NSL_KDD99/KDDTrain+.txt", 
            "data/NSL_KDD99/KDDTest+.txt"
        )
        
        # Analyze labels to understand the data
        print("\n🔍 Analyzing data distribution...")
        train_label_analysis = loader.analyze_labels(train_df)
        test_label_analysis = loader.analyze_labels(test_df)
        
        # 🎯 Step 2: Create binary labels using label 21 as normal (most frequent)
        print("\n🎯 Phase 2: Creating binary classification labels...")
        train_df = loader.create_binary_labels(train_df, normal_label=21)
        test_df = loader.create_binary_labels(test_df, normal_label=21)
        
        # Check if we have a reasonable class distribution
        normal_count_train = (train_df['is_attack'] == 0).sum()
        normal_count_test = (test_df['is_attack'] == 0).sum()
        
        print(f"\n📊 Final Class Distribution:")
        print(f"   Training - Normal: {normal_count_train:,}, Attack: {len(train_df)-normal_count_train:,}")
        print(f"   Testing  - Normal: {normal_count_test:,}, Attack: {len(test_df)-normal_count_test:,}")
        
        if normal_count_train == 0:
            print("❌ No normal samples found! Trying alternative labels...")
            # Try the second most frequent label as normal
            second_most_frequent = train_label_analysis.index[1]
            print(f"   Trying label {second_most_frequent} as normal...")
            train_df = loader.create_binary_labels(train_df, normal_label=second_most_frequent)
            test_df = loader.create_binary_labels(test_df, normal_label=second_most_frequent)
        
        # 🔧 Step 3: Preprocess Data
        print("\n🔧 Phase 3: Preprocessing Network Data...")
        preprocessor = CyberPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.preprocess_features(
            train_df, test_df
        )
        
        # Check class distribution in processed data
        unique_train = np.unique(y_train)
        unique_test = np.unique(y_test)
        print(f"\n🔍 Class check - Train: {unique_train}, Test: {unique_test}")
        
        # 🤖 Step 4: Initialize Models
        print("\n🤖 Phase 4: Deploying BlackWall ML Models...")
        model_manager = BlackWallModels()
        models = model_manager.initialize_models()
        
        # 📊 Step 5: Train & Evaluate
        print("\n📊 Phase 5: Training & Evaluation...")
        evaluator = CyberEvaluator()
        
        successful_models = 0
        
        for name, model in models.items():
            print(f"   Training {name}...")
            try:
                if name == 'IsolationForest':
                    # IsolationForest is unsupervised
                    model.fit(X_train)
                    y_pred = model.predict(X_test)
                    y_pred = (y_pred == -1).astype(int)  # Convert to binary
                else:
                    # Supervised models
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                results = evaluator.evaluate_model(name, model, X_test, y_test, y_pred)
                print(f"   ✅ {name} - F1 Score: {results['f1_score']:.3f}")
                successful_models += 1
                
            except Exception as e:
                print(f"   ❌ {name} failed: {str(e)}")
                continue
        
        # 🎨 Step 6: Visualize Results
        if successful_models > 0:
            print(f"\n🎨 Phase 6: Generating Cyber Dashboard ({successful_models} models successful)...")
            try:
                evaluator.plot_performance_dashboard()
            except Exception as e:
                print(f"   ⚠️  Visualization skipped: {str(e)}")
        
        # 🏆 Display Final Results
        print("\n🏆 BLACKWALL DEPLOYMENT COMPLETE")
        print("═" * 50)
        if evaluator.results:
            print("🔒 MODEL PERFORMANCE SUMMARY:")
            for model_name, results in evaluator.results.items():
                print(f"   {model_name:20} | F1: {results['f1_score']:.3f} | "
                      f"Accuracy: {results['accuracy']:.3f} | "
                      f"Precision: {results['precision']:.3f} | Recall: {results['recall']:.3f}")
            
            # Find best model
            best_model = max(evaluator.results.items(), key=lambda x: x[1]['f1_score'])
            print(f"\n🎯 BEST MODEL: {best_model[0]} (F1: {best_model[1]['f1_score']:.3f})")
        else:
            print("❌ No models were successfully trained")
            print("\n💡 Troubleshooting tips:")
            print("   1. Check if you have both normal and attack samples")
            print("   2. Try different normal labels (run discovery script)")
            print("   3. Ensure data files are correct NSL-KDD format")
    
    except Exception as e:
        print(f"\n💥 CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()