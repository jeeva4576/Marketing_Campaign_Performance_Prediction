# %%
import pickle
import os
import streamlit as st
import pandas as pd
from scipy.sparse import hstack

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marketing Campaign Performance Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

output_folder = (
    r"D:\Data Science\vscode\Marketing_Campaign_Performance_Prediction"
    r"\Output"
)

master_data_path = os.path.join(
    output_folder,
    "Marketing_Campaign_Feature_Engineered_Master.csv"
)

regression_model_path = os.path.join(
    output_folder,
    "regression_gradient_boosting_model.pkl"
)

classification_model_path = os.path.join(
    output_folder,
    "classification_gradient_boosting_model.pkl"
)

regression_encoder_path = os.path.join(
    output_folder,
    "regression_encoder.pkl"
)

classification_encoder_path = os.path.join(
    output_folder,
    "classification_encoder.pkl"
)


# ============================================================
# LOAD MASTER DATASET
# ============================================================

@st.cache_data
def load_master_data():

    return pd.read_csv(master_data_path)


df = load_master_data()


# ============================================================
# INTERNAL DEFAULTS
# Required model features not shown in the UI
# ============================================================

duration_default = df["Duration"].median()

date_series = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

year_default = int(
    date_series.dt.year.mode()[0]
)

month_default = int(
    date_series.dt.month.mode()[0]
)

dayofweek_default = int(
    date_series.dt.dayofweek.mode()[0]
)


# ============================================================
# LOAD TRAINED MODELS AND ENCODERS
# ============================================================

@st.cache_resource
def load_models():

    with open(regression_model_path, "rb") as file:
        regression_model = pickle.load(file)

    with open(classification_model_path, "rb") as file:
        classification_model = pickle.load(file)

    with open(regression_encoder_path, "rb") as file:
        regression_encoder = pickle.load(file)

    with open(classification_encoder_path, "rb") as file:
        classification_encoder = pickle.load(file)

    return (
        regression_model,
        classification_model,
        regression_encoder,
        classification_encoder
    )


(
    regression_model,
    classification_model,
    regression_encoder,
    classification_encoder
) = load_models()


# ============================================================
# TITLE
# ============================================================

st.title(
    "📊 Marketing Campaign Performance Prediction"
)

st.write(
    "Predict campaign Revenue or Profit/Loss using "
    "Gradient Boosting Machine Learning models."
)

st.divider()


# ============================================================
# MODEL SELECTION
# ============================================================

st.header("🤖 Select Prediction Model")

prediction_type = st.radio(
    "Choose the prediction you want:",
    [
        "Regression — Revenue Prediction",
        "Classification — Profit/Loss Prediction"
    ],
    horizontal=True
)

st.divider()


# ============================================================
# CAMPAIGN INPUTS
# ============================================================

st.header("📋 Campaign Inputs")


col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

with col1:

    campaign_type = st.selectbox(
        "Campaign Type",
        sorted(
            df["Campaign_Type"]
            .dropna()
            .unique()
        )
    )

    target_audience = st.selectbox(
        "Target Audience",
        sorted(
            df["Target_Audience"]
            .dropna()
            .unique()
        )
    )

    impressions = st.number_input(
        "Impressions",
        min_value=0,
        value=10000,
        step=100
    )

    clicks = st.number_input(
        "Clicks",
        min_value=0,
        value=1000,
        step=10
    )


# ============================================================
# COLUMN 2
# ============================================================

with col2:

    channel_options = [
    "Google",
    "WhatsApp",
    "YouTube",
    "Email",
    "Instagram",
    "Facebook"
]

    selected_channels = st.multiselect(
    "Channel Used",
    options=channel_options,
    default=["Email"],
    max_selections=6
    )

    leads = st.number_input(
        "Leads",
        min_value=0,
        value=100,
        step=1
    )

    conversions = st.number_input(
        "Conversions",
        min_value=0,
        value=20,
        step=1
    )

    acquisition_cost = st.number_input(
        "Acquisition Cost",
        min_value=1.0,
        value=1000.0,
        step=100.0
    )


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    calculated_roi = st.number_input(
        "Calculated ROI",
        value=1.0,
        step=0.01,
        format="%.2f"
    )

    engagement_score = st.number_input(
        "Engagement Score",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    language = st.selectbox(
        "Language",
        sorted(
            df["Language"]
            .dropna()
            .unique()
        )
    )

    customer_segment = st.selectbox(
        "Customer Segment",
        sorted(
            df["Customer_Segment"]
            .dropna()
            .unique()
        )
    )


# ============================================================
# BRAND
# ============================================================

brand = st.selectbox(
    "Brand",
    sorted(
        df["Company_Name"]
        .dropna()
        .unique()
    )
)


# ============================================================
# CHANNEL FEATURES
# Convert selected individual channels into
# the original model representation
# ============================================================

channel_order = [
    "Email",
    "Facebook",
    "Google",
    "Instagram",
    "WhatsApp",
    "YouTube"
]

selected_channels_sorted = sorted(
    selected_channels,
    key=channel_order.index
)

channel_used = ", ".join(
    selected_channels_sorted
)


# ============================================================
# CHANNEL INDICATORS
# ============================================================

channel_indicators = {}

for channel in channel_order:

    channel_indicators[
        f"Channel_{channel}"
    ] = int(
        channel in selected_channels
    )


# ============================================================
# MODEL STATUS
# ============================================================

st.divider()

if prediction_type == "Regression — Revenue Prediction":

    st.info(
        "📈 Regression model selected — Revenue will be predicted."
    )

else:

    st.info(
        "🎯 Classification model selected — "
        "Profit/Loss will be predicted."
    )


# ============================================================
# PREDICT BUTTON
# IMPORTANT:
# Both regression and classification prediction logic is inside
# this button block. Therefore, no prediction is displayed
# until the user clicks Predict.
# ============================================================

st.divider()

if st.button(
    "🔮 Predict",
    use_container_width=True
):

    st.header("📊 Prediction Result")


    # ========================================================
    # REGRESSION — REVENUE PREDICTION
    # ========================================================

    if prediction_type == "Regression — Revenue Prediction":

        regression_input = pd.DataFrame([{

            "Duration": duration_default,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "Acquisition_Cost": acquisition_cost,
            "Engagement_Score": engagement_score,
            "Calculated_ROI": calculated_roi,

            "Channel_Google":
                channel_indicators["Channel_Google"],
            "Channel_WhatsApp":
                channel_indicators["Channel_WhatsApp"],
            "Channel_YouTube":
                channel_indicators["Channel_YouTube"],
            "Channel_Email":
                channel_indicators["Channel_Email"],
            "Channel_Instagram":
                channel_indicators["Channel_Instagram"],
            "Channel_Facebook":
                channel_indicators["Channel_Facebook"],

            "Year": year_default,
            "Month": month_default,
            "DayOfWeek": dayofweek_default,

            "Campaign_Type": campaign_type,
            "Target_Audience": target_audience,
            "Channel_Used": channel_used,
            "Language": language,
            "Customer_Segment": customer_segment,
            "Company_Name": brand

        }])


        # ====================================================
        # REGRESSION FEATURES
        # These match the frozen Gradient Boosting model.
        # ====================================================

        regression_numerical_features = [
            "Duration",
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions",
            "Acquisition_Cost",
            "Engagement_Score",
            "Calculated_ROI",
            "Channel_Google",
            "Channel_WhatsApp",
            "Channel_YouTube",
            "Channel_Email",
            "Channel_Instagram",
            "Channel_Facebook",
            "Year",
            "Month",
            "DayOfWeek"
        ]

        regression_categorical_features = [
            "Campaign_Type",
            "Target_Audience",
            "Channel_Used",
            "Language",
            "Customer_Segment",
            "Company_Name"
        ]


        # ====================================================
        # PREPARE REGRESSION FEATURES
        # ====================================================

        X_numeric = regression_input[
            regression_numerical_features
        ].values

        X_categorical = regression_input[
            regression_categorical_features
        ]

        X_categorical_encoded = (
            regression_encoder.transform(
                X_categorical
            )
        )


        # ====================================================
        # COMBINE FEATURES
        # ====================================================

        X_regression = hstack([
            X_numeric,
            X_categorical_encoded
        ])


        # ====================================================
        # REVENUE PREDICTION
        # ====================================================

        prediction = regression_model.predict(
            X_regression
        )[0]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        st.success(
            f"💰 Predicted Revenue: ₹{prediction:,.2f}"
        )


    # ========================================================
    # CLASSIFICATION — PROFIT / LOSS PREDICTION
    # ========================================================

    else:

        classification_input = pd.DataFrame([{

            "Duration": duration_default,
            "Impressions": impressions,
            "Clicks": clicks,
            "Leads": leads,
            "Conversions": conversions,
            "Acquisition_Cost": acquisition_cost,
            "Calculated_ROI": calculated_roi,
            "Engagement_Score": engagement_score,

            "Channel_Google":
                channel_indicators["Channel_Google"],
            "Channel_WhatsApp":
                channel_indicators["Channel_WhatsApp"],
            "Channel_YouTube":
                channel_indicators["Channel_YouTube"],
            "Channel_Email":
                channel_indicators["Channel_Email"],
            "Channel_Instagram":
                channel_indicators["Channel_Instagram"],
            "Channel_Facebook":
                channel_indicators["Channel_Facebook"],

            "Campaign_Type": campaign_type,
            "Target_Audience": target_audience,
            "Channel_Used": channel_used,
            "Language": language,
            "Customer_Segment": customer_segment,
            "Company_Name": brand

        }])


        # ====================================================
        # CLASSIFICATION FEATURES
        # These match the approved classification model.
        # ====================================================

        classification_numerical_features = [
            "Duration",
            "Impressions",
            "Clicks",
            "Leads",
            "Conversions",
            "Acquisition_Cost",
            "Calculated_ROI",
            "Engagement_Score",
            "Channel_Google",
            "Channel_WhatsApp",
            "Channel_YouTube",
            "Channel_Email",
            "Channel_Instagram",
            "Channel_Facebook"
        ]

        classification_categorical_features = [
            "Campaign_Type",
            "Target_Audience",
            "Channel_Used",
            "Language",
            "Customer_Segment",
            "Company_Name"
        ]


        # ====================================================
        # PREPARE CLASSIFICATION FEATURES
        # ====================================================

        X_numeric = classification_input[
            classification_numerical_features
        ].values

        X_categorical = classification_input[
            classification_categorical_features
        ]

        X_categorical_encoded = (
            classification_encoder.transform(
                X_categorical
            )
        )


        # ====================================================
        # COMBINE FEATURES
        # ====================================================

        X_classification = hstack([
            X_numeric,
            X_categorical_encoded
        ])


        # ====================================================
        # CLASSIFICATION PREDICTION
        # ====================================================

        prediction = classification_model.predict(
            X_classification
        )[0]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if prediction == 1:

            st.success(
                "✅ Predicted Result: PROFIT"
            )

        else:

            st.error(
                "⚠️ Predicted Result: LOSS"
            )

# %%
