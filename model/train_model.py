import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# load dataset
df = pd.read_csv('../data/customers.csv')

# features (X) and target (y)
X = df[['age', 'purchases']]
y = df['churn']

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y)

# train model
model = LogisticRegression()
model.fit(X_train, y_train)

# save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")