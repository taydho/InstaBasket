import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to calculate user features
def user_features(data):
    user_orders = data.groupby('user_id')['order_number'].max().reset_index()
    user_orders.columns = ['user_id', 'user_orders']

    user_period = data.groupby('user_id')['days_since_prior_order'].sum().reset_index()
    user_period.columns = ['user_id', 'user_period']

    user_mean_days_since_prior_order = data.groupby('user_id')['days_since_prior_order'].mean().reset_index()
    user_mean_days_since_prior_order.columns = ['user_id', 'user_mean_days_since_prior_order']

    # Calculate reorder ratio for each user
    user_total_orders = data.groupby('user_id')['order_id'].count().reset_index()
    user_total_orders.columns = ['user_id', 'user_total_orders']

    user_total_reorders = data.groupby('user_id')['reordered'].sum().reset_index()
    user_total_reorders.columns = ['user_id', 'user_total_reorders']

    user_reorder_ratio = user_total_reorders['user_total_reorders'] / user_total_orders['user_total_orders']
    user_reorder_ratio_df = pd.concat([user_total_orders['user_id'], user_reorder_ratio], axis=1)
    user_reorder_ratio_df.columns = ['user_id', 'user_reorder_ratio']

    # Merge reorder ratio with other user features
    user_features_df = pd.merge(user_orders, user_period, on='user_id')
    user_features_df = pd.merge(user_features_df, user_mean_days_since_prior_order, on='user_id')
    user_features_df = pd.merge(user_features_df, user_reorder_ratio_df, on='user_id')

    return user_features_df


# Function to calculate product features
def product_features(data):
    product_orders = data.groupby('product_id')['order_id'].count().reset_index()
    product_orders.columns = ['product_id', 'product_orders']

    product_reorders = data.groupby('product_id')['reordered'].sum().reset_index()
    product_reorders.columns = ['product_id', 'product_reorders']

    product_reorder_rate = product_reorders['product_reorders'] / product_orders['product_orders']
    product_features_df = pd.concat([product_orders, product_reorders['product_reorders'], product_reorder_rate], axis=1)
    product_features_df.columns = ['product_id', 'product_orders', 'product_reorders', 'product_reorder_rate']

    return product_features_df


# Function to calculate user-product interaction features
def user_product_interaction_features(data):
    user_product_orders = data.groupby(['user_id', 'product_id'])['order_id'].count().reset_index()
    user_product_orders.columns = ['user_id', 'product_id', 'user_product_orders']

    user_product_avg_add_to_cart_order = data.groupby(['user_id', 'product_id'])['add_to_cart_order'].mean().reset_index()
    user_product_avg_add_to_cart_order.columns = ['user_id', 'product_id', 'user_product_avg_add_to_cart_order']

    user_product_features_df = pd.merge(user_product_orders, user_product_avg_add_to_cart_order, on=['user_id', 'product_id'])

    return user_product_features_df


def calculate_aisle_department_proportions(prior_data):
    # Group the data by user_id and aisle_id and calculate the size (i.e., count of rows) of each group
    user_aisle_counts = prior_data.groupby(['user_id', 'aisle_id']).size().reset_index(name='count')

    # Group the data by user_id and department_id and calculate the size (i.e., count of rows) of each group
    user_department_counts = prior_data.groupby(['user_id', 'department_id']).size().reset_index(name='count')

    # Calculate the total count of rows for each user
    user_total_counts = prior_data.groupby('user_id').size().reset_index(name='total')

    # Merge the aisle count and total count dataframes on user_id
    user_aisle_proportions = user_aisle_counts.merge(user_total_counts, on='user_id')

    # Calculate the proportion of aisle orders for each user by dividing aisle count by total count
    user_aisle_proportions['proportion'] = user_aisle_proportions['count'] / user_aisle_proportions['total']

    # Merge the department count and total count dataframes on user_id
    user_department_proportions = user_department_counts.merge(user_total_counts, on='user_id')

    # Calculate the proportion of department orders for each user by dividing department count by total count
    user_department_proportions['proportion'] = user_department_proportions['count'] / user_department_proportions['total']

    # Return the aisle and department proportions dataframes
    return user_aisle_proportions, user_department_proportions

def create_features(prior_data, train_data, products_df, aisles_df, departments_df):
    # Extract user features from the train_data
    user_features_train = user_features(train_data)

    # Extract product features from the train_data
    product_features_train = product_features(train_data)

    # Extract user-product interaction features from the train_data
    # user_product_interaction_features_train = user_product_interaction_features(train_data)

    # Extract aisle and department proportions for each user from the prior_data
    # user_aisle_proportions, user_department_proportions = calculate_aisle_department_proportions(prior_data)

    # Merge user, product, user-product interaction, aisle and department features with the train_data
    train_data = pd.merge(train_data, user_features_train, on='user_id')
    train_data = pd.merge(train_data, product_features_train, on='product_id')
    # train_data = pd.merge(train_data, user_product_interaction_features_train, on=['user_id', 'product_id'])
    # train_data = pd.merge(train_data, user_aisle_proportions, on='user_id', how='left')
    # train_data = pd.merge(train_data, user_department_proportions, on='user_id', how='left')

    # Drop irrelevant columns from the train_data
    train_data = train_data.drop(columns=['user_id', 'user_mean_days_since_prior_order', 'days_since_prior_order', 'add_to_cart_order'])

    # Drop non-numerical columns from the train_data
    train_data = train_data.select_dtypes(include=['number'])

    # Prepare the features for the prior_data by dropping the reordered column
    model_features = prior_data.drop(columns=['reordered'])

    # Prepare the features and labels for the train_data
    features = train_data.drop(columns=['reordered'])
    labels = train_data['reordered'].values

    # Split the train_data into training, validation, and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2, random_state=42)
    train_features, val_features, train_labels, val_labels = train_test_split(train_features, train_labels, test_size=0.25, random_state=42)

    # Drop rows with missing values from the train_features and val_features
    train_mask = train_features.notna().all(axis=1)
    train_features = train_features[train_mask]
    train_labels = train_labels[train_mask]

    val_mask = val_features.notna().all(axis=1)
    val_features = val_features[val_mask]
    val_labels = val_labels[val_mask]

    # Return the extracted features and labels for training, validation, and test sets
    return model_features, train_features, train_labels, val_features, val_labels, test_features, test_labels
