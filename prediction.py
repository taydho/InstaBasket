import pandas as pd
from feature_engineering import user_features, product_features

def make_predictions(model, test_features):
    # Make predictions
    predictions = model.predict(test_features)

    # Convert predictions to a 1-dimensional NumPy array
    predictions_np = predictions.reshape(-1)

    # Get the unique order_ids
    unique_order_ids = test_features['order_id'].unique()

    # Check if lengths match and if not, truncate the longer one
    if len(unique_order_ids) != len(predictions_np):
        min_length = min(len(unique_order_ids), len(predictions_np))
        unique_order_ids = unique_order_ids[:min_length]
        predictions_np = predictions_np[:min_length]

    # Save predictions to output.csv
    output = pd.DataFrame({'order_id': unique_order_ids, 'products': predictions_np})
    output.to_csv('output.csv', index=False)

    return predictions


def preprocess_and_extract_features(test_data):
    # Preprocess test data and create test features similar to feature_engineering.py
    user_features_test = user_features(test_data)
    product_features_test = product_features(test_data)

    test_features = pd.merge(test_data, user_features_test, on='user_id')
    test_features = pd.merge(test_features, product_features_test, on='product_id')

    # Drop non-numerical columns
    test_features = test_features.select_dtypes(include=['number'])

    # Extract test_order_ids
    test_order_ids = test_data['order_id'].unique()

    return test_features, test_order_ids
