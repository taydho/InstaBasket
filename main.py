from data_preparation import load_and_preprocess_data
from feature_engineering import create_features
from train import train_model, grid_search, perform_cross_validation
from prediction import make_predictions
from xgboost import XGBClassifier
import time

def main():
    prior_data, train_data = load_and_preprocess_data()
    model_features, train_features, train_labels, val_features, val_labels, test_features, test_labels = create_features(prior_data, train_data)
    
    # Perform grid search
    best_model = grid_search(train_features, train_labels)

    # Perform cross-validation
    perform_cross_validation(best_model, train_features, train_labels)

    # Train the model with the best hyperparameters
    model = train_model(train_features, train_labels, val_features, val_labels, best_model)

    make_predictions(model, test_features)  # Pass test_features as an argument

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training completed in {duration // 60:.0f}m {duration % 60:.0f}s')