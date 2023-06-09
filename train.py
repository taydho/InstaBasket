from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score, RandomizedSearchCV, KFold
from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer


def train_xgb_with_early_stopping(xgb_model, train_features, train_labels, val_features, val_labels):
    """Trains XGBoost model with early stopping."""
    xgb_model.fit(train_features, train_labels)
    return xgb_model


def perform_cross_validation(model, features, labels, k_folds=5, scoring='roc_auc'):
    """Performs cross-validation and prints mean and standard deviation of the score."""
    kf = KFold(n_splits=k_folds)
    scores = []

    for train_index, val_index in kf.split(features):
        X_train, X_val = features.iloc[train_index], features.iloc[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        fit_params = {
            'verbose': False
        }

        model.fit(X_train, y_train, **fit_params)

        if scoring == 'roc_auc':
            y_pred = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred)
        else:
            y_pred = model.predict(X_val)
            score = accuracy_score(y_val, y_pred)

        scores.append(score)

    mean_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Cross-validation {scoring}: {mean_score:.4f} (+/- {std_score:.4f})")


def train_model(train_features, train_labels, val_features, val_labels, xgb_model): 
    """Trains a stacking ensemble classifier using XGBoost and Histogram-based Gradient Boosting."""
    log_reg_clf = HistGradientBoostingClassifier(random_state=42, max_iter=1000)
    meta_classifier = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

    # Define the base estimators
    base_estimators = [
        ('xgb', xgb_model),
        ('lr', log_reg_clf)
    ]
    # Instantiate the StackingClassifier
    ensemble = StackingClassifier(estimators=base_estimators, final_estimator=meta_classifier)
    # Use mean strategy for imputing missing values
    imputer = SimpleImputer(strategy='mean')
    train_features = imputer.fit_transform(train_features)
    val_features = imputer.transform(val_features)

    # Fit the ensemble to the training data
    ensemble.fit(train_features, train_labels)

    # Make predictions on the validation set
    val_preds = ensemble.predict(val_features)
    val_probs = ensemble.predict_proba(val_features)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds)
    recall = recall_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds)
    auc_roc = roc_auc_score(val_labels, val_probs)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")

    return ensemble
