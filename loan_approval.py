
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load dataset
data = pd.read_csv("loan_data.csv")

# Encode categorical columns
le = LabelEncoder()
for col in ["Gender", "Married", "Education", "Loan_Status"]:
    data[col] = le.fit_transform(data[col])

# Split features and target
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# ---------------------------------
# NEW PERSON LOAN PREDICTION
# ---------------------------------

# Example new applicant details
# Gender: Male=1 Female=0
# Married: Yes=1 No=0
# Education: Graduate=1 Not Graduate=0

new_person = [[
    1,   # Gender (Male)
    1,   # Married (Yes)
    1,   # Education (Graduate)
    5000, # Applicant Income
    150,  # Loan Amount
    1     # Credit History
]]

# Scale new person data
new_person_scaled = scaler.transform(new_person)

# Predict loan status
result = model.predict(new_person_scaled)

# Display result
if result[0] == 1:
    print("Loan Status: APPROVED ✅")
else:
    print("Loan Status: NOT APPROVED ❌")
