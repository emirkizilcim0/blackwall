import pandas as pd
import numpy as np

# Quick check of your data
columns = [
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

print("🔍 DEBUG: Checking NSL-KDD Data Distribution...")

train_df = pd.read_csv("data/NSL_KDD99/KDDTrain+.txt", names=columns)
test_df = pd.read_csv("data/NSL_KDD99/KDDTest+.txt", names=columns)

print(f"📊 Training data label distribution:")
print(train_df['label'].value_counts().head(10))

print(f"\n📊 Test data label distribution:")
print(test_df['label'].value_counts().head(10))

print(f"\n🎯 Binary classification check:")
train_df['is_attack'] = (train_df['label'] != 'normal').astype(int)
test_df['is_attack'] = (test_df['label'] != 'normal').astype(int)

print(f"Training - Normal: {(train_df['is_attack'] == 0).sum():,}, Attack: {(train_df['is_attack'] == 1).sum():,}")
print(f"Test - Normal: {(test_df['is_attack'] == 0).sum():,}, Attack: {(test_df['is_attack'] == 1).sum():,}")

# Check if 'normal' exists in training data
if 'normal' not in train_df['label'].values:
    print("\n❌ CRITICAL: No 'normal' samples in training data!")
    print("   This is why supervised models are failing.")
    print("   Possible solutions:")
    print("   1. Use different dataset files")
    print("   2. Check if you're using the correct NSL-KDD files")
    print("   3. The training file might be corrupted or wrong version")
else:
    print("\n✅ 'normal' samples found in training data")