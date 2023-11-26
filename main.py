import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
matplotlib.rcParams['font.sans-serif'] = 'Arial'

# Disable PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
data = pd.read_csv('/Users/amadoudiakhadiop/Documents/pythonProject6/Expresso_churn_dataset.csv')


numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Handle missing values for numerical columns
for col in numerical_cols:
    data[col].fillna(data[col].mean(), inplace=True)  # Replace missing values with the mean

# Handle missing values for categorical columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)  # Replace missing values with the mode

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical column
data_encoded = data.copy()
for col in categorical_cols:
    data_encoded[col] = label_encoder.fit_transform(data_encoded[col])

# Streamlit App
st.title('Expresso Churn Analysis')

# Data Overview
st.subheader('Data Overview')
st.write(data_encoded.head())

# Visualizations
st.subheader('Visualizations')

# Region-wise Churn
st.subheader('Region-wise Churn')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='REGION', hue='CHURN', data=data, ax=ax)
st.pyplot(fig)

# Tenure vs. Churn
st.subheader('Tenure vs. Churn')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='CHURN', y='TENURE', data=data, ax=ax)
st.pyplot(fig)

# Top Products and Churn
st.subheader('Top Products and Churn')
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(x='TOP_PACK', hue='CHURN', data=data, order=data['TOP_PACK'].value_counts().index, ax=ax)
plt.xticks(rotation=45, ha='right')  # Modify the figure directly
st.pyplot(fig)

# Income (Revenue) and Churn (Histogram or Kernel Density Plot)
st.subheader('Income (Revenue) and Churn (Histogram or Kernel Density Plot)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(x='REVENUE', hue='CHURN', data=data, kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Customer Regularity and Churn (Histogram or Kernel Density Plot)
st.subheader('Customer Regularity and Churn (Histogram or Kernel Density Plot)')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(x='REGULARITY', hue='CHURN', data=data, kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Frequency of Recharges vs. Churn (Scatter Plot or Boxplot)
st.subheader('Frequency of Recharges vs. Churn (Scatter Plot or Boxplot)')
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(x='CHURN', y='FREQUENCE_RECH', data=data, ax=ax)
st.pyplot(fig)

# Usage of Different Services (On_NET, Orange, Tigo) and Churn (Stacked Bar Plots)
services = ['ON_NET', 'ORANGE', 'TIGO']
data_services = data[services + ['CHURN']].copy()

# Melt the DataFrame for easier plotting
data_services_melted = data_services.melt(id_vars=['CHURN'], value_vars=services)

st.subheader('Usage of Different Services (On_NET, Orange, Tigo) and Churn (Stacked Bar Plots)')
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='variable', hue='CHURN', y='value', data=data_services_melted, errorbar=None, ax=ax)
ax.set_xlabel('Services')  # Set x-axis label directly on Axes object
ax.set_ylabel('Proportion')
st.pyplot(fig)


features = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK']
target_variable = 'CHURN'

# Selecting features and target variable
X = data_encoded[features]
y = data_encoded[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to evaluate
models = {
'Neural Network': MLPClassifier()}

# Results dictionary to store model performances
results = {'Model': [], 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': []}

# Evaluate each model
for model_name, model in models.items():
    # Cross-validate the model
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(scores)

    # Additional metrics
    y_pred = model.fit(X_train_scaled, y_train).predict(X_test_scaled)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # Store the results
    results['Model'].append(model_name)
    results['Accuracy'].append(mean_accuracy)
    results['Precision'].append(precision)
    results['Recall'].append(recall)
    results['F1-Score'].append(f1)

    # Display confusion matrix
    st.subheader('Confusion Matrix')
    sns.heatmap(confusion_mat, annot=True, fmt='g')
    st.pyplot()

# Display all results
results_df = pd.DataFrame(results)
st.subheader('Model Results')
st.write(results_df)
st.subheader('DONE!')

