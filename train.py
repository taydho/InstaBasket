
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import KFold

def perform_cross_validation(model, features, labels, k_folds=5, scoring='roc_auc'):
    kf = KFold(n_splits=k_folds)
    scores = []

    for train_index, val_index in kf.split(features):
        X_train, X_val = features.iloc[train_index], features.iloc[val_index]
        y_train, y_val = labels[train_index], labels[val_index]

        # Set early_stopping_rounds using set_params
        model.set_params(early_stopping_rounds=10)
        
        fit_params = {
            'eval_set': [(X_val, y_val)],
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

def grid_search(train_features, train_labels, n_iter=50):  # n_iter is the number of parameter settings sampled
    # Define the hyperparameters search space
    param_dist = {
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
        'gamma': [0, 0.1, 0.2]
    }

    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        early_stopping_rounds=10,
        n_jobs=-1,
        use_label_encoder=False
    )

    # Instantiate the randomized search object
    randomized_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='roc_auc',
        cv=3,
        verbose=1,
        n_jobs=-1
    )

    # Define fit_params for early stopping and evaluation set
    fit_params = {
            'eval_set': [(train_features, train_labels)]
        }

    # Fit the randomized search object to the training data
    randomized_search.fit(train_features, train_labels, **fit_params)

    # Print the best hyperparameters
    print("Best hyperparameters:", randomized_search.best_params_)

    return randomized_search.best_estimator_


def train_model(train_features, train_labels, val_features, val_labels, best_model):

    # Train the model with the best hyperparameters
    best_model.fit(train_features, train_labels, eval_set=[(val_features, val_labels)])

    # Make predictions on the validation set
    val_preds = best_model.predict(val_features)
    val_probs = best_model.predict_proba(val_features)[:, 1]

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


    return best_model
