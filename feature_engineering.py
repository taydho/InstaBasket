import pandas as pd

def product_popularity(df):
    popularity = df['product_id'].value_counts().reset_index()
    popularity.columns = ['product_id', 'popularity']
    return popularity

def create_features(prior_data, train_data):
    # Calculate product popularity
    prior_popularity = product_popularity(prior_data)
    train_popularity = product_popularity(train_data)

    # Merge product popularity with the original data
    prior_data = pd.merge(prior_data, prior_popularity, on='product_id')
    train_data = pd.merge(train_data, train_popularity, on='product_id')

    # Select the features we want to use
    prior_features = prior_data[['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'popularity']]
    train_features = train_data[['order_dow', 'order_hour_of_day', 'days_since_prior_order', 'popularity']]

    return prior_features, train_features
