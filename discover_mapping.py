import pandas as pd
import numpy as np

# Let's discover what the numeric labels actually mean
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

print("🔍 DISCOVERING NSL-KDD NUMERIC MAPPINGS...")

train_df = pd.read_csv("data/NSL_KDD99/KDDTrain+.txt", names=columns)
test_df = pd.read_csv("data/NSL_KDD99/KDDTest+.txt", names=columns)

print("📊 UNIQUE VALUES ANALYSIS:")

print("\n🎯 LABEL column:")
print("Unique values:", sorted(train_df['label'].unique()))
print("Value counts:")
print(train_df['label'].value_counts().sort_index())

print("\n🔌 PROTOCOL_TYPE column:")
print("Unique values:", sorted(train_df['protocol_type'].unique()))
print("Value counts:")
print(train_df['protocol_type'].value_counts().sort_index())

print("\n🌐 SERVICE column:")
print("Unique values:", sorted(train_df['service'].unique())[:20])  # First 20
print("Total unique services:", train_df['service'].nunique())

print("\n🚩 FLAG column:")
print("Unique values:", sorted(train_df['flag'].unique()))
print("Value counts:")
print(train_df['flag'].value_counts().sort_index())

print("\n💡 SUGGESTED MAPPING BASED ON FREQUENCY:")
print("Most frequent label (potential 'normal'):", train_df['label'].mode()[0])
print("Most frequent protocol:", train_df['protocol_type'].mode()[0])
print("Most frequent service:", train_df['service'].mode()[0])
print("Most frequent flag:", train_df['flag'].mode()[0])