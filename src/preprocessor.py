from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class CyberPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def preprocess_features(self, train_df, test_df, target_col='is_attack'):
        """Preprocess features for ML models - robust to mixed data types"""
        print("🔧 Starting preprocessing...")
        
        # Identify feature columns (exclude target and label)
        exclude_cols = [target_col, 'label']
        feature_columns = [col for col in train_df.columns if col not in exclude_cols]
        
        print(f"   Using {len(feature_columns)} feature columns")
        
        # Separate features and target
        X_train = train_df[feature_columns].copy()
        X_test = test_df[feature_columns].copy()
        y_train = train_df[target_col]
        y_test = test_df[target_col]
        
        print(f"   Original shapes - Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Identify categorical columns (non-numeric)
        categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        
        print(f"   Numeric columns: {len(numeric_cols)}")
        print(f"   Categorical columns: {categorical_cols}")
        
        # Handle categorical features with one-hot encoding
        if categorical_cols:
            X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols)
            X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols)
            
            print(f"   After encoding - Train: {X_train_encoded.shape}, Test: {X_test_encoded.shape}")
            
            # Align columns
            X_train_final, X_test_final = X_train_encoded.align(
                X_test_encoded, join='left', axis=1, fill_value=0
            )
        else:
            X_train_final = X_train.copy()
            X_test_final = X_test.copy()
        
        # Ensure all data is numeric
        X_train_final = X_train_final.apply(pd.to_numeric, errors='coerce')
        X_test_final = X_test_final.apply(pd.to_numeric, errors='coerce')
        
        # Fill any remaining NaN values with 0
        X_train_final = X_train_final.fillna(0)
        X_test_final = X_test_final.fillna(0)
        
        print(f"   Final shapes - Train: {X_train_final.shape}, Test: {X_test_final.shape}")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_final)
        X_test_scaled = self.scaler.transform(X_test_final)
        
        # Store feature columns for reference
        self.feature_columns = X_train_final.columns.tolist()
        
        print("✅ Preprocessing complete!")
        
        return X_train_scaled, X_test_scaled, y_train, y_test