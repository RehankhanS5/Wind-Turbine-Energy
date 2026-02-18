import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# =========================
# 1. LOAD DATASET
# =========================
data = pd.read_csv("data/wind_turbine.csv")

# =========================
# 2. CLEAN COLUMN NAMES
# =========================
data.columns = (
    data.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("/", "_per_", regex=False)
    .str.replace("°", "deg", regex=False)
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

print("Columns after cleaning:")
print(data.columns)

# =========================
# 3. DROP DATE/TIME COLUMN
# =========================
if "Date_per_Time" in data.columns:
    data.drop(columns=["Date_per_Time"], inplace=True)
    print("Dropped column: Date_per_Time")

# =========================
# 4. SELECT FEATURES (MATCH WEB FORM)
# =========================
FEATURES = [
    "Wind_Speed_m_per_s",
    "Wind_Direction_deg",
    "Air_Density",
    "Temperature",
    "Humidity",
    "Blade_Length"
]

TARGET = "Power_Output"

# =========================
# 5. KEEP ONLY REQUIRED COLUMNS
# =========================
data = data[FEATURES + [TARGET]]

# Convert everything to numeric
data = data.apply(pd.to_numeric, errors="coerce")
data.dropna(inplace=True)

print("\nFinal Columns Used:")
print(data.columns)

# =========================
# 6. SPLIT FEATURES & TARGET
# =========================
X = data[FEATURES]
y = data[TARGET]

# =========================
# 7. TRAIN MODEL
# =========================
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X, y)

# =========================
# 8. SAVE MODEL
# =========================
joblib.dump(model, "wind_model.pkl")

print("\n✅ Model trained and saved as wind_model.pkl")
