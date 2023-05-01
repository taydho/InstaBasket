from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def train_xgb_with_early_stopping(xgb_model, train_features, train_labels, val_features, val_labels):
    xgb_model.fit(train_features, train_labels)
    return xgb_model




def perform_cross_validation(model, features, labels, k_folds=5, scoring='roc_auc'):
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

    # Logistic Regression classifier
    clf2 = LogisticRegression(random_state=42)

    # Define the base estimators
    base_estimators = [
        ('xgb', xgb_model),
        ('lr', clf2)
    ]

    # Define the meta-classifier
    meta_classifier = LogisticRegression(random_state=42)

    # Instantiate the StackingClassifier
    ensemble = StackingClassifier(estimators=base_estimators, final_estimator=meta_classifier)

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
