import pickle
from schema.patient_schema import PatientRecord
from adapters.manual_adapter import ManualEntryAdapter

# Load your trained model + encoder
xgb_model = pickle.load(open('models/xgb_model.pkl', 'rb'))
le = pickle.load(open('models/label_encoder.pkl', 'rb'))

# Simulate a form submission
adapter = ManualEntryAdapter()
records = adapter.parse({
    'hemoglobin': 7.2, 'platelets': 65000,
    'INR': 1.9, 'age': 58, 'surgery_type': 'Emergency'
})

# Convert to model input and predict
features = records[0].to_model_features(le)
prediction = xgb_model.predict([features])[0]
print('Transfusion needed:', bool(prediction))