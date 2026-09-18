import streamlit as st
import sys
import os
import shap
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)


# ============================================================
# IMPORT PREDICTION FUNCTIONS
# ============================================================

from src.prediction import (
    model,
    preprocessor,
    FEATURES,
    create_transaction_features,
    predict_transaction
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔐",
    layout="wide"
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "🔐 AI-Powered Fraud Detection & "
    "Transaction Risk Analysis System"
)

st.write(
    "A machine learning system that analyzes financial "
    "transactions and identifies potentially fraudulent "
    "activity using Random Forest and Explainable AI."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("ℹ️ About the System")

    st.write(
        "This application uses a Random Forest machine "
        "learning model trained on the PaySim dataset."
    )

    st.write("### Technologies")

    st.write(
        """
        • Python  
        • Pandas  
        • Scikit-learn  
        • Random Forest  
        • SHAP  
        • Streamlit  
        """
    )

    st.write("### Risk Levels")

    st.write(
        """
        🟢 LOW — Fraud probability below 30%

        🟡 MEDIUM — Fraud probability from 30% to below 70%

        🔴 HIGH — Fraud probability 70% or above
        """
    )

    st.info(
        "These risk thresholds are project-defined "
        "and are not universal banking standards."
    )


# ============================================================
# TRANSACTION INPUT
# ============================================================

st.header("💳 Transaction Information")

st.write(
    "Enter the transaction details below."
)

col1, col2 = st.columns(2)


with col1:

    step = st.number_input(
        "Transaction Step",
        min_value=1,
        value=1,
        step=1,
        help="Time step of the transaction in the PaySim dataset."
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    oldbalanceOrg = st.number_input(
        "Sender Balance Before Transaction",
        min_value=0.0,
        value=10000.0,
        step=100.0
    )


with col2:

    newbalanceOrig = st.number_input(
        "Sender Balance After Transaction",
        min_value=0.0,
        value=9000.0,
        step=100.0
    )

    oldbalanceDest = st.number_input(
        "Receiver Balance Before Transaction",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    newbalanceDest = st.number_input(
        "Receiver Balance After Transaction",
        min_value=0.0,
        value=6000.0,
        step=100.0
    )


st.divider()


# ============================================================
# ANALYZE TRANSACTION
# ============================================================

if st.button(
    "🔍 Analyze Transaction",
    use_container_width=True,
    type="primary"
):

    transaction = {

        "step": step,

        "type": transaction_type,

        "amount": amount,

        "oldbalanceOrg": oldbalanceOrg,

        "newbalanceOrig": newbalanceOrig,

        "oldbalanceDest": oldbalanceDest,

        "newbalanceDest": newbalanceDest
    }


    # ========================================================
    # PREDICTION
    # ========================================================

    result = predict_transaction(transaction)

    prediction = result["prediction"]

    probability = result["fraud_probability"]

    risk = result["risk"]


    st.divider()

    st.header("📊 Transaction Analysis Result")


    # ========================================================
    # RESULT STATUS
    # ========================================================

    if prediction == 1:

        st.error(
            "🚨 Potentially Fraudulent Transaction"
        )

        st.write(
            "The model classified this transaction "
            "as potentially fraudulent."
        )

    else:

        st.success(
            "✅ Transaction Classified as Legitimate"
        )

        st.write(
            "The model classified this transaction "
            "as legitimate."
        )


    # ========================================================
    # METRICS
    # ========================================================

    result_col1, result_col2, result_col3 = st.columns(3)


    with result_col1:

        st.metric(
            "Fraud Probability",
            f"{probability * 100:.2f}%"
        )


    with result_col2:

        st.metric(
            "Risk Level",
            risk
        )


    with result_col3:

        st.metric(
            "Prediction",
            "FRAUD" if prediction == 1
            else "LEGITIMATE"
        )


    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.subheader("Fraud Probability")

    st.progress(
        min(max(probability, 0.0), 1.0)
    )


    # ========================================================
    # TRANSACTION SUMMARY
    # ========================================================

    st.subheader("📋 Transaction Summary")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Transaction Step",
            "Transaction Type",
            "Amount",
            "Sender Balance Before",
            "Sender Balance After",
            "Receiver Balance Before",
            "Receiver Balance After"
        ],

        "Value": [
            step,
            transaction_type,
            f"{amount:,.2f}",
            f"{oldbalanceOrg:,.2f}",
            f"{newbalanceOrig:,.2f}",
            f"{oldbalanceDest:,.2f}",
            f"{newbalanceDest:,.2f}"
        ]
    })

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🧠 Explainable AI — SHAP Analysis"
    )

    st.write(
        "SHAP helps explain which model features "
        "influenced this prediction."
    )


    try:

        transaction_df = pd.DataFrame(
            [transaction]
        )

        transaction_df = create_transaction_features(
            transaction_df
        )

        model_input = transaction_df[
            FEATURES
        ]

        processed_input = preprocessor.transform(
            model_input
        )


        # ----------------------------------------------------
        # SHAP EXPLAINER
        # ----------------------------------------------------

        explainer = shap.TreeExplainer(
            model
        )

        shap_values = explainer.shap_values(
            processed_input
        )


        # ----------------------------------------------------
        # HANDLE SHAP OUTPUT FORMAT
        # ----------------------------------------------------

        if isinstance(shap_values, list):

            shap_values_for_transaction = (
                shap_values[1][0]
            )

        elif len(shap_values.shape) == 3:

            shap_values_for_transaction = (
                shap_values[0, :, 1]
            )

        else:

            shap_values_for_transaction = (
                shap_values[0]
            )


        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        # ----------------------------------------------------
        # SHAP DATAFRAME
        # ----------------------------------------------------

        explanation_df = pd.DataFrame({

            "Feature": feature_names,

            "SHAP Impact":
                shap_values_for_transaction
        })


        explanation_df[
            "Absolute Impact"
        ] = explanation_df[
            "SHAP Impact"
        ].abs()


        explanation_df = (
            explanation_df
            .sort_values(
                "Absolute Impact",
                ascending=False
            )
            .head(10)
        )


        # ----------------------------------------------------
        # SHAP CHART
        # ----------------------------------------------------

        chart_df = (
            explanation_df
            .sort_values(
                "Absolute Impact"
            )
        )


        fig, ax = plt.subplots(
            figsize=(10, 5)
        )


        ax.barh(
            chart_df["Feature"],
            chart_df["SHAP Impact"]
        )


        ax.set_xlabel(
            "SHAP Impact"
        )


        ax.set_ylabel(
            "Feature"
        )


        ax.set_title(
            "Top Features Influencing the Prediction"
        )


        plt.tight_layout()


        st.pyplot(fig)

        plt.close(fig)


        # ----------------------------------------------------
        # SHAP TABLE
        # ----------------------------------------------------

        st.subheader(
            "Top Influencing Features"
        )


        display_df = explanation_df[
            [
                "Feature",
                "SHAP Impact"
            ]
        ].copy()


        display_df[
            "SHAP Impact"
        ] = display_df[
            "SHAP Impact"
        ].round(4)


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )


        st.info(
            "Positive SHAP values push the model "
            "towards the fraud class, while negative "
            "values push the model towards the "
            "legitimate class."
        )


    except Exception as e:

        st.warning(
            "SHAP explanation could not be generated."
        )

        st.write(
            "Technical details:",
            str(e)
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Powered Fraud Detection & Transaction Risk Analysis "
    "System | Random Forest + SHAP + Streamlit"
)