"""Central configuration: paths, schema and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
RAW_PATH = DATA_DIR / "telco.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "avg_monthly_spend",
                    "n_services"]
BINARY_FEATURES = ["SeniorCitizen", "gender", "Partner", "Dependents", "PhoneService",
                   "PaperlessBilling"]
CATEGORICAL_FEATURES = ["MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                        "Contract", "PaymentMethod", "tenure_band"]

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]

# Business assumptions used to pick the decision threshold.
# A retention offer costs £X; a saved customer is worth ~12 months of revenue.
RETENTION_OFFER_COST = 50.0
RETENTION_SUCCESS_RATE = 0.30
CUSTOMER_VALUE_MONTHS = 12
