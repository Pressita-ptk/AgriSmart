import joblib

# Load trained model
model = joblib.load("model.pkl")

# Example values from the dataset
sample = [[90, 42, 43, 20.87, 82.0, 6.5, 202.9]]

prediction = model.predict(sample)

print("Predicted Crop:", prediction[0])