# Ansh Sharma (12509226)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\anshs\OneDrive\Documents\EXCEL FILES\IndusBank BS .csv").dropna()

# Target variable
y = df['TAD']

# Feature variables
X = df.drop(columns=['TAD']).select_dtypes(include='number')

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree Regressor model
model = DecisionTreeRegressor(max_depth=4, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2:", r2_score(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)

# Plot Decision Tree
plt.figure(figsize=(8, 6))
plot_tree(
    model,
    feature_names=X.columns,
    filled=True,
    fontsize=6
)

plt.tight_layout()
plt.show()