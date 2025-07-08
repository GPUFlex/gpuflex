import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import torch

housing = fetch_california_housing()
X = housing.data
y = housing.target

df = pd.DataFrame(X, columns=housing.feature_names)
df['target'] = y
df.to_csv("california_housing.csv", index=False)


df = pd.read_csv("california_housing.csv")

X = df.drop(columns=["target"]).values
y = df["target"].values.reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

dataset = (X_tensor, y_tensor)