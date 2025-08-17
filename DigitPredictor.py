from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=2000).fit(X_train, y_train)

# Test model on unseen data
print("Actual:", y_test[0])
print("Predicted:", model.predict([X_test[0]])[0])

