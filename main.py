# Import necessary modules and functions
from data_preparation import load_and_preprocess_data
from feature_engineering import create_features
from train import train_model, perform_cross_validation, train_xgb_with_early_stopping
from prediction import make_predictions
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the main function
def main():
    # Load and preprocess the data
    prior_data, train_data, products_df, aisles_df, departments_df = load_and_preprocess_data()
    
    # Create features from the data
    model_features, train_features, train_labels, val_features, val_labels, test_features, test_labels = create_features(prior_data, train_data, products_df, aisles_df, departments_df)

    # Initialize an XGBoost model and train it with early stopping
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
    xgb_model = train_xgb_with_early_stopping(xgb_model, train_features, train_labels, val_features, val_labels)

    # # Plot feature importances for the XGBoost model
    # feature_importances = xgb_model.feature_importances_
    # importance_df = pd.DataFrame({'feature': train_features.columns, 'importance': feature_importances})
    # importance_df = importance_df.sort_values('importance', ascending=False)
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='importance', y='feature', data=importance_df)
    # plt.title('Feature Importances')
    # plt.xlabel('Importance')
    # plt.ylabel('Feature')
    # plt.show()

    # Scale the training and validation data
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

    # Train an ensemble model using the scaled data
    trained_ensemble = train_model(train_features_scaled, train_labels, val_features_scaled, val_labels, xgb_model)

    # Make predictions with the trained ensemble model
    make_predictions(trained_ensemble, test_features)

if __name__ == "__main__":
    # Record the start time
    start_time = time.time()

    # Call the main function
    main()

    # Calculate the total training time and print it
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training completed in {duration // 60:.0f}m {duration % 60:.0f}s')
