import os
import joblib
import pandas as pd


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_paysim_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_paysim_preprocessor.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "fraud_detection_features.pkl"
)


# --------------------------------------------------
# LOAD MODEL COMPONENTS
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
FEATURES = joblib.load(FEATURES_PATH)


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def create_transaction_features(transaction):
    transaction = transaction.copy()

    transaction["orig_balance_change"] = (
        transaction["oldbalanceOrg"]
        - transaction["newbalanceOrig"]
    )

    transaction["dest_balance_change"] = (
        transaction["newbalanceDest"]
        - transaction["oldbalanceDest"]
    )

    transaction["amount_to_oldbalance_ratio"] = (
        transaction["amount"]
        / (transaction["oldbalanceOrg"] + 1)
    )

    transaction["amount_to_dest_balance_ratio"] = (
        transaction["amount"]
        / (transaction["oldbalanceDest"] + 1)
    )

    transaction["orig_balance_error"] = (
        transaction["oldbalanceOrg"]
        - transaction["amount"]
        - transaction["newbalanceOrig"]
    )

    transaction["dest_balance_error"] = (
        transaction["oldbalanceDest"]
        + transaction["amount"]
        - transaction["newbalanceDest"]
    )

    return transaction


# --------------------------------------------------
# PREPARE TRANSACTION
# --------------------------------------------------

def prepare_transaction(transaction):
    transaction_df = pd.DataFrame([transaction])

    transaction_df = create_transaction_features(
        transaction_df
    )

    return transaction_df[FEATURES]


# --------------------------------------------------
# RISK LEVEL
# --------------------------------------------------

def get_risk_level(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


# --------------------------------------------------
# PREDICT TRANSACTION
# --------------------------------------------------

def predict_transaction(transaction):

    model_input = prepare_transaction(transaction)

    processed_input = preprocessor.transform(
        model_input
    )

    prediction = int(
        model.predict(processed_input)[0]
    )

    fraud_probability = float(
        model.predict_proba(processed_input)[0][1]
    )

    risk = get_risk_level(
        fraud_probability
    )

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "risk": risk
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_transaction = {
        "step": 100,
        "type": "TRANSFER",
        "amount": 25000,
        "oldbalanceOrg": 50000,
        "newbalanceOrig": 25000,
        "oldbalanceDest": 10000,
        "newbalanceDest": 35000
    }

    result = predict_transaction(
        test_transaction
    )

    print("\nTransaction Prediction")
    print("----------------------")
    print("Prediction:", result["prediction"])
    print(
        "Fraud Probability:",
        f"{result['fraud_probability'] * 100:.2f}%"
    )
    print("Risk Level:", result["risk"])