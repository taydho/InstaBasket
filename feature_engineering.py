import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def user_features(data):
    user_orders = data.groupby('user_id')['order_number'].max().reset_index()
    user_orders.columns = ['user_id', 'user_orders']

    user_period = data.groupby('user_id')['days_since_prior_order'].sum().reset_index()
    user_period.columns = ['user_id', 'user_period']

    user_mean_days_since_prior_order = data.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    user_mean_days_since_prior_order.columns = ['user_id', 'user_mean_days_since_prior_order']

    user_features_df = pd.merge(user_orders, user_period, on='user_id')
    user_features_df = pd.merge(user_features_df, user_mean_days_since_prior_order, on='user_id')

    return user_features_df


def product_features(data):
    product_orders = data.groupby('product_id')['order_id'].count().reset_index()
    product_orders.columns = ['product_id', 'product_orders']

    product_reorders = data.groupby('product_id')['reordered'].sum().reset_index()
    product_reorders.columns = ['product_id', 'product_reorders']

    product_reorder_rate = product_reorders['product_reorders'] / product_orders['product_orders']
    product_features_df = pd.concat([product_orders, product_reorders['product_reorders'], product_reorder_rate], axis=1)
    product_features_df.columns = ['product_id', 'product_orders', 'product_reorders', 'product_reorder_rate']

    return product_features_df

def create_features(prior_data, train_data):
    user_features_train = user_features(train_data)
    product_features_train = product_features(train_data)

    train_data = pd.merge(train_data, user_features_train, on='user_id')
    train_data = pd.merge(train_data, product_features_train, on='product_id')

    # Drop non-numerical columns
    train_data = train_data.select_dtypes(include=['number'])

    # Prepare the features for the prior_data
    model_features = prior_data.drop(columns=['reordered'])

    # Prepare the features and labels for the train_data
    features = train_data.drop(columns=['reordered'])
    labels = train_data['reordered'].values

    # Split the train_data into training, validation, and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.25, random_state=42)

    return model_features, train_features, train_labels, val_features, val_labels, test_features, test_labels
