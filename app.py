import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("student-por.csv")

# Features and target
X = df[['studytime','failures','absences','G1','G2']]
y = df['G3']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("🎓 Student Exam Score Predictor")

st.write("This app predicts the final exam score (G3) based on studytime, failures, absences, and previous grades (G1, G2).")

studytime = st.number_input("Study Time (hours/week)", min_value=1, max_value=20)
failures = st.number_input("Past Failures", min_value=0, max_value=5)
absences = st.number_input("Absences", min_value=0, max_value=50)
g1 = st.number_input("First Period Grade (G1)", min_value=0, max_value=20)
g2 = st.number_input("Second Period Grade (G2)", min_value=0, max_value=20)

if st.button("Predict Final Score"):
    input_data = np.array([[studytime, failures, absences, g1, g2]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Final Exam Score: {prediction[0]:.2f}")

# Show model performance
st.subheader("📊 Model Performance")
st.write(f"MAE: {mae:.2f}")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.2f}")
