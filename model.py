import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model and scaler
with open('fuel_effi.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

st.title("Fuel Efficiency Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:")
    st.write(df)

    # Ensure the 'UTC' column exists and convert it to datetime format
    if 'UTC' in df.columns:
        df['UTC'] = pd.to_datetime(df['UTC'], unit='s')
    else:
        st.error("UTC column not found in the uploaded CSV file.")

    # Rename columns to match those expected by the model
    df.rename(columns={'HRlFC': 'HRLFC'}, inplace=True)

    # Calculate fuel efficiency using the provided code
    if 'TotalDistance' in df.columns and 'HRLFC' in df.columns:
        df['TotalDistance'] = df['TotalDistance'].diff()
        df['fuel'] = df['HRLFC'].diff()
        df = df.dropna()
        df['fuel_efficiency'] = df['TotalDistance'] / df['fuel']
        df = df.dropna()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=['fuel_efficiency'], inplace=True)
    else:
        st.error("TotalDistance or HRLFC columns not found in the uploaded CSV file.")

    # Check for missing values
    st.write("Missing values in the uploaded data:")
    st.write(df.isna().sum())

    # Calculate idling time where the engine speed is > 0 and vehicle speed = 0
    if 'EngineSpeed' in df.columns and 'VehicleSpeed' in df.columns:
        df['idling_time'] = np.where((df['EngineSpeed'] > 0) & (df['VehicleSpeed'] == 0), df['UTC'].diff().dt.total_seconds(), 0)
    else:
        st.error("EngineSpeed or VehicleSpeed columns not found in the uploaded CSV file.")

    # # Plot graphs
    # if 'VehicleSpeed' in df.columns and 'fuel_efficiency' in df.columns:
    #     st.write("Fuel Efficiency vs Vehicle Speed")
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(data=df, x='VehicleSpeed', y='fuel_efficiency', ax=ax)
    #     plt.xlabel('Vehicle Speed')
    #     plt.ylabel('Fuel Efficiency')
    #     st.pyplot(fig)
    # else:
    #     st.error("VehicleSpeed or fuel_efficiency columns not found in the DataFrame.")

    # if 'EngineCoolantTemp' in df.columns and 'fuel_efficiency' in df.columns:
    #     st.write("Fuel Efficiency vs Engine Coolant Temperature")
    #     fig, ax = plt.subplots()
    #     sns.scatterplot(data=df, x='EngineCoolantTemp', y='fuel_efficiency', ax=ax)
    #     plt.xlabel('Engine Coolant Temperature')
    #     plt.ylabel('Fuel Efficiency')
    #     st.pyplot(fig)
    # else:
    #     st.error("EngineCoolantTemp or fuel_efficiency columns not found in the DataFrame.")

    # Check for the required columns before plotting the heatmap
    required_columns = ['VehicleSpeed', 'AccPedalPosition', 'TotalDistance', 'EngineCoolantTemp', 'EngineOilPressure','fuel_efficiency']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if not missing_columns:
        st.write("Heatmap of Correlations")
        corr = df[required_columns].corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.error(f"Missing columns for correlation heatmap: {', '.join(missing_columns)}")

# Manual input for prediction
st.write("Manual Input for Prediction")
vehicle_speed = st.number_input("Vehicle Speed", value=0)
acc_pedal_position = st.number_input("Acceleration Pedal Position", value=0.0)
acceleration = st.number_input("Acceleration", value=0.0)
total_distance = st.number_input("Total Distance", value=0.0)
engine_coolant_temp = st.number_input("Engine Coolant Temperature", value=0.0)
engine_oil_pressure = st.number_input("Engine Oil Pressure", value=0.0)
hrlfc = st.number_input("HRLFC", value=0.0)

if st.button("Predict Fuel Efficiency"):
    input_data = pd.DataFrame({
        'VehicleSpeed': [vehicle_speed],
        'AccPedalPosition': [acc_pedal_position],
        'acceleration': [acceleration],
        'TotalDistance': [total_distance],
        'EngineCoolantTemp': [engine_coolant_temp],
        'EngineOilPressure': [engine_oil_pressure],
        'HRLFC': [hrlfc]
    })

    # Debug print input data
    st.write("Input Data:")
    st.write(input_data)
    
    input_data_scaled = scaler.transform(input_data)
    
    # Debug print scaled input data
    st.write("Scaled Input Data:")
    st.write(input_data_scaled)
    
    prediction = model.predict(input_data_scaled)
    
    # Debug print prediction
    st.write("Prediction:")
    st.write(prediction)
    
    st.write(f"Predicted Fuel Efficiency: {prediction[0]} km/unit")

