from data_preparation import load_and_preprocess_data
from feature_engineering import create_features
from train import train_model, perform_cross_validation, train_xgb_with_early_stopping
from prediction import make_predictions
from xgboost import XGBClassifier
import time

def main():
    prior_data, train_data = load_and_preprocess_data()
    model_features, train_features, train_labels, val_features, val_labels, test_features, test_labels = create_features(prior_data, train_data)
    
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        colsample_bytree=0.5,
        gamma=0,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=1,
        n_estimators=300,
        subsample=1,
        eval_metric='logloss',
        n_jobs=-1,
        use_label_encoder=False,
        seed=42
    )


    # Perform cross-validation
    perform_cross_validation(xgb_model, train_features, train_labels)

    # Train XGBoost model with early stopping
    xgb_model = train_xgb_with_early_stopping(xgb_model, train_features, train_labels, val_features, val_labels)

    # Train the model with the best hyperparameters
    model = train_model(train_features, train_labels, val_features, val_labels, xgb_model)

    make_predictions(model, test_features)  # Pass test_features as an argument

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training completed in {duration // 60:.0f}m {duration % 60:.0f}s')
