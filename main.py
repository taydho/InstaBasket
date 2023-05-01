from data_preparation import load_and_preprocess_data
from feature_engineering import create_features
from model import train_and_evaluate_model
from prediction import make_predictions

# Load and preprocess data
prior_data, train_data = load_and_preprocess_data()

# Create features
prior_features, train_features = create_features(prior_data, train_data)

# Train and evaluate the model
model, evaluation_metrics = train_and_evaluate_model(train_features)

# Make predictions for new orders
new_data = pd.DataFrame()  # Replace with actual new data
predictions = make_predictions(model, new_data)
