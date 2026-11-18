import pandas as pd
import pytest

from churn import config as C
from churn.data import clean, validate
from churn.features import FeatureBuilder, build_preprocessor


@pytest.fixture
def raw():
    return pd.DataFrame([{
        "customerID": "1", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 0, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "Yes", "StreamingTV": "No",
        "StreamingMovies": "No", "Contract": "Two year", "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check", "MonthlyCharges": 50.0, "TotalCharges": " ",
        "Churn": "No"}])


def test_blank_total_charges_for_new_customers_become_zero(raw):
    validate(raw)
    out = clean(raw)
    assert out.loc[0, "TotalCharges"] == 0.0


def test_validate_rejects_missing_columns(raw):
    with pytest.raises(ValueError, match="Missing columns"):
        validate(raw.drop(columns=["tenure"]))


def test_feature_builder_derives_expected_columns(raw):
    X = FeatureBuilder().transform(clean(raw).drop(columns=["customerID", "Churn"]))
    assert X.loc[0, "n_services"] == 3  # Phone, OnlineSecurity, TechSupport
    assert X.loc[0, "tenure_band"] == "0-6m"
    assert X.loc[0, "avg_monthly_spend"] == 50.0  # falls back to MonthlyCharges at tenure 0


def test_preprocessor_handles_unseen_categories(raw):
    X = clean(raw).drop(columns=["customerID", "Churn"])
    pre = build_preprocessor().fit(X)
    unseen = X.copy()
    unseen["PaymentMethod"] = "Crypto"
    assert pre.transform(unseen).shape == pre.transform(X).shape


def test_feature_lists_are_disjoint():
    groups = [set(C.NUMERIC_FEATURES), set(C.BINARY_FEATURES), set(C.CATEGORICAL_FEATURES)]
    assert sum(map(len, groups)) == len(set().union(*groups))
