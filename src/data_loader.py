import pandas as pd
import numpy as np

class NSLKDDLoader:
    def __init__(self):
        self.columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', 'label'
        ]
        
        # Based on your data discovery, label 21 is the most frequent (potential 'normal')
        self.normal_label = 21
    
    def load_data(self, train_path, test_path):
        """Load NSL-KDD dataset"""
        print("📥 Loading dataset files...")
        train_df = pd.read_csv(train_path, names=self.columns)
        test_df = pd.read_csv(test_path, names=self.columns)
        
        print(f"   Raw Data Loaded:")
        print(f"   Training samples: {len(train_df):,}")
        print(f"   Testing samples: {len(test_df):,}")
        
        return train_df, test_df
    
    def create_binary_labels(self, df, normal_label=None):
        """Create binary labels: normal vs attack"""
        if normal_label is None:
            normal_label = self.normal_label
        
        print(f"🎯 Creating binary labels using label {normal_label} as 'normal'...")
        
        # Convert label to string for consistent handling
        df['label'] = df['label'].astype(str)
        
        # Create binary classification
        df['is_attack'] = (df['label'] != str(normal_label)).astype(int)
        
        attack_count = df['is_attack'].sum()
        normal_count = len(df) - attack_count
        
        print(f"   Normal samples (label {normal_label}): {normal_count:,}")
        print(f"   Attack samples: {attack_count:,}")
        print(f"   Attack ratio: {attack_count/len(df):.3f}")
        
        # Show top 10 labels for verification
        print(f"   Top 10 labels: {df['label'].value_counts().head(10).to_dict()}")
        
        return df
    
    def analyze_labels(self, df):
        """Analyze label distribution to help identify normal traffic"""
        print("\n🔍 Label Analysis:")
        label_counts = df['label'].value_counts().sort_index()
        
        for label, count in label_counts.head(15).items():
            print(f"   Label {label}: {count:>6,} samples")
        
        # Suggest potential normal labels (those with high frequency)
        potential_normal = label_counts.head(3).index.tolist()
        print(f"💡 Suggested normal labels (most frequent): {potential_normal}")
        
        return label_counts