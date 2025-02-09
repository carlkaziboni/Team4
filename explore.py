"""
This is where all the models will be stored. They will run at the beginning of the flask app
"""

import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
import csv
from sklearn.preprocessing import LabelEncoder

data_diageo_path = os.getcwd() +"/Diageo_Scotland_Full_Year_2024_Daily_Data.csv"
data_diageo_df = pd.read_csv(data_diageo_path)
data_diageo_df

data_diageo_numeriical_df = data_diageo_df.drop(labels=["Timestamp", "Site", "Scope_1_Emissions_tonnes_CO2e", "Scope_2_Emissions_tonnes_CO2e"], axis=1)
data_diageo_numeriical_df

data_diageo_numeriical_mean = data_diageo_numeriical_df.mean()

data_diageo_numerical_std = data_diageo_numeriical_df.std()

data_diageo_numeriical_standardized_df = (data_diageo_numeriical_df - data_diageo_numeriical_mean) / (data_diageo_numerical_std)
data_diageo_numeriical_covariance = data_diageo_numeriical_standardized_df.cov()

eigenvalues, eigenvectors = np.linalg.eig(data_diageo_numeriical_covariance)

idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

explained_var = np.cumsum(eigenvalues)/np.sum(eigenvalues)

n_components = np.argmax(explained_var >= 0.95) +1


pca = PCA(n_components=n_components)
pca.fit(data_diageo_numeriical_standardized_df)
x_pca = pca.transform(data_diageo_numeriical_standardized_df)

df_pca1 = pd.DataFrame(x_pca,
                       columns=['PC{}'.
                       format(i+1)
                        for i in range(n_components)])

pca_df = pd.DataFrame(x_pca[:, :7], columns=[f'PC{i+1}' for i in range(7)])

y_1 = data_diageo_df['Scope_1_Emissions_tonnes_CO2e']
y_2 = data_diageo_df['Scope_2_Emissions_tonnes_CO2e']
X_train, X_test, y_train, y_test = train_test_split(x_pca, y_1, test_size=0.3, random_state=42)
model_PCA_y_1 = LinearRegression()
model_PCA_y_1.fit(X_train, y_train)

model_PCA_Y_2 = LinearRegression()
model_PCA_Y_2.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(x_pca, y_2, test_size=0.3, random_state=42)


X = data_diageo_df.drop(["Scope_1_Emissions_tonnes_CO2e", 'Scope_2_Emissions_tonnes_CO2e', "Timestamp", "Site"], axis=1)
Y_1 = data_diageo_df['Scope_1_Emissions_tonnes_CO2e']
Y_2 = data_diageo_df['Scope_2_Emissions_tonnes_CO2e']

X_train, X_test, y_train, y_test = train_test_split(X, Y_1, test_size=0.3, random_state=42)
model_multi_y1 = LinearRegression()
model_multi_y1.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y_2, test_size=0.3, random_state=42)
model_multi_y2 = LinearRegression()
model_multi_y2.fit(X_train, y_train)


X_train, X_test, y_train, y_test = train_test_split(X, Y_1, test_size=0.3, random_state=42)
model_MLP_y1 = MLPRegressor(max_iter=500)
model_MLP_y1.fit(X_train, y_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y_2, test_size=0.3, random_state=42)
model_MLP_y2 = MLPRegressor(max_iter=500)
model_MLP_y2.fit(X_train, y_train)

# Read CSV and extract features/labels
with open("Diageo_Scotland_Full_Year_2024_Daily_Data.csv") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header

    data = []
    for row in reader:
        data.append({
            "evidence": [float(cell) for cell in row[2:4]] + [float(cell) for cell in row[6:]],
            "y_1": row[4]  # Ensure this is properly converted
        })

# Convert evidence and labels
evidence = [row["evidence"] for row in data]
y_1 = [row["y_1"] for row in data]

# Convert y_1 to numeric values (check if it's categorical)
try:
    y_1 = [float(y) for y in y_1]  # If y_1 is numerical
except ValueError:
    encoder = LabelEncoder()
    y_1 = encoder.fit_transform(y_1)  # Convert categorical labels to integers

X_training, X_testing, y_training, y_testing = train_test_split(
    np.array(evidence), np.array(y_1), test_size=0.4, random_state=42
)

# Define neural network model
model_neural_network = tf.keras.models.Sequential([
    tf.keras.layers.Dense(9, input_shape=(9,), activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")  # Keep if binary classification
])

# Compile the model
model_neural_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss="binary_crossentropy", metrics=["accuracy"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)  # Use same scaler


# Convert to tensors
X_tensor = tf.convert_to_tensor(X_training, dtype=tf.float32)
Y_tensor = tf.convert_to_tensor(y_training, dtype=tf.float32)

# Train model
model_neural_network.fit(X_tensor, Y_tensor, epochs=200, batch_size=32, validation_split=0.1)



class ModelsMade():
    def PCAfunc():
        return model_PCA_y_1, model_PCA_Y_2
    
    def MultiLinear():
        return model_multi_y1, model_multi_y2
    
    def MLPModel():
        return model_MLP_y1, model_MLP_y2
    
    def NNModel():
        return model_neural_network
    
print(ModelsMade.PCAfunc())