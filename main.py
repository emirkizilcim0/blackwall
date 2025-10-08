# main.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("data/NSL_KDD99/train.csv")

# Display sample
print(df.head())

# Encode categorical features (e.g., protocol_type, service, flag)
categorical_cols = ['protocol_type', 'service', 'flag']
df = pd.get_dummies(df, columns=categorical_cols)

# Encode labels
label_enc = LabelEncoder()
df['label'] = label_enc.fit_transform(df['label'])

# Split data
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train baseline model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
