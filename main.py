import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import load_data, basic_cleaning
from src.modeling import train_logistic_model
from src.evaluation import evaluate_model

# Step 1: Load and Clean Data
df = load_data('data/LoanTapData.csv')
df = basic_cleaning(df)

# Step 2: Preprocessing
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# Step 3: Split Features and Target
X = df.drop('loan_status', axis=1)
y = df['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Model Training
model = train_logistic_model(X_train_scaled, y_train)

# Step 6: Evaluation
evaluate_model(model, X_test_scaled, y_test)
