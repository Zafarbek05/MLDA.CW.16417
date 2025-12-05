import joblib
import pandas as pd
import streamlit as st

# 1. Define Constants and Paths
LAGGED_MOBILITY_INPUTS = [
    'retail_and_recreation_percent_change_from_baseline_roll7_lag7',
    'grocery_and_pharmacy_percent_change_from_baseline_roll7_lag7',
    'parks_percent_change_from_baseline_roll7_lag7',
    'transit_stations_percent_change_from_baseline_roll7_lag7',
    'workplaces_percent_change_from_baseline_roll7_lag7',
    'residential_percent_change_from_baseline_roll7_lag7'
]

# Decoder for One_Hot Encoded Features
OHE_STATES = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
              'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho',
              'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana',
              'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
              'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada',
              'New Hampshire', 'New Jersey', 'New Mexico', 'New York',
              'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon',
              'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota',
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington',
              'West Virginia', 'Wisconsin', 'Wyoming']


OHE_DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

OHE_MONTHS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

# 2. Load Assets
MODEL_PATH = 'joblibs/F1_final_mlr_model.joblib'
SCALER_PATH = 'joblibs/F1_feature_scaler.joblib'
FEATURE_LIST_PATH = 'joblibs/F1_feature_list.joblib'

@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_cols = joblib.load(FEATURE_LIST_PATH)
        return model, scaler, feature_cols
    except FileNotFoundError:
        st.error(f"Required deployment files (e.g., {MODEL_PATH}) not found. "
                 "Please ensure the training script was run successfully "
                 "and save the Linear Regression model and preprocessing files.")
        return None, None, None


model, scaler, FEATURE_COLS = load_assets()


# 3. Preprocessing
def preprocess_and_predict(raw_inputs, model, scaler, feature_cols):
    X_predict = pd.DataFrame(0, index=[0], columns=feature_cols)
    state_col = f'sub_region_1_{raw_inputs["state"]}'
    if state_col in X_predict.columns:
        X_predict.loc[0, state_col] = 1

    # Day Mapping (using day_of_week)
    day_col = f'day_of_week_{raw_inputs["day"]}'
    if day_col in X_predict.columns:
        X_predict.loc[0, day_col] = 1

    # Month Mapping
    month_col = f'month_{raw_inputs["month"]}'
    if month_col in X_predict.columns:
        X_predict.loc[0, month_col] = 1

    # Extract the raw mobility values in the correct order
    raw_mobility_values = [raw_inputs[col] for col in LAGGED_MOBILITY_INPUTS]

    # Scale the extracted values using the loaded scaler
    scaled_mobility_values = scaler.transform([raw_mobility_values])[0]

    # Insert the scaled values back into the X_predict DataFrame
    for i, col in enumerate(LAGGED_MOBILITY_INPUTS):
        X_predict.loc[0, col] = scaled_mobility_values[i]

    prediction = model.predict(X_predict)

    return max(0, prediction[0])


# 4. Streamlit UI
st.set_page_config(
    page_title="COVID-19 Case Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🦠 COVID-19 Case Rate Forecast Application")
st.subheader("Predicting Weekly New Cases using Mobility Data")

if model is not None:
    st.markdown("""
    This application simulates the deployment of the final **Linear Regression** model.
    Input the mean mobility values for a specific state and time point.
    """)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Contextual Features")
            selected_state = st.selectbox(
                "State/Region",
                options=OHE_STATES,
                index=OHE_STATES.index('New York') if 'New York' in OHE_STATES else 0
            )
            selected_month = st.selectbox(
                "Month",
                options=OHE_MONTHS,
                index=3
            )
            selected_day = st.selectbox(
                "Day of the Week",
                options=OHE_DAYS,
                index=0
            )

        with col2:
            st.markdown("### Mobility Inputs (percentage from baseline)")

            retail = st.number_input("Retail & Recreation (% Change from Baseline)", min_value=-100.0, max_value=100.0,
                                     value=-5.0, step=0.1)
            grocery = st.number_input("Grocery & Pharmacy (% Change from Baseline)", min_value=-100.0, max_value=100.0,
                                      value=2.0, step=0.1)
            parks = st.number_input("Parks (% Change from Baseline)", min_value=-100.0, max_value=200.0, value=15.0,
                                    step=0.1)
            transit = st.number_input("Transit Stations (% Change from Baseline)", min_value=-100.0, max_value=100.0,
                                      value=-12.0, step=0.1)
            workplaces = st.number_input("Workplaces (% Change from Baseline)", min_value=-100.0, max_value=100.0,
                                         value=-18.0, step=0.1)
            residential = st.number_input("Residential (% Change from Baseline)", min_value=-100.0, max_value=100.0,
                                          value=4.0, step=0.1)

        submitted = st.form_submit_button("Forecast Weekly New Cases")

    if submitted:
        raw_inputs = {
            "state": selected_state,
            "month": selected_month,
            "day": selected_day,
            'retail_and_recreation_percent_change_from_baseline_roll7_lag7': retail,
            'grocery_and_pharmacy_percent_change_from_baseline_roll7_lag7': grocery,
            'parks_percent_change_from_baseline_roll7_lag7': parks,
            'transit_stations_percent_change_from_baseline_roll7_lag7': transit,
            'workplaces_percent_change_from_baseline_roll7_lag7': workplaces,
            'residential_percent_change_from_baseline_roll7_lag7': residential
        }

        try:
            with st.spinner('Calculating prediction...'):
                prediction = preprocess_and_predict(raw_inputs, model, scaler, FEATURE_COLS)

            st.success("✅ Prediction Complete")

            st.metric(
                label=f"Predicted Weekly New Cases for {selected_state}",
                value=f"{int(prediction):,}"
            )

            st.info(
                f"The model has a Root Mean Squared Error (RMSE) of **17,482.35** on the test set, meaning the prediction can typically deviate by this much.")

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.warning("Please check that the feature list and column names match the training script exactly.")