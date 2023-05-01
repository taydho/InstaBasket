import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_data():
    aisles = pd.read_csv('./instacart-market-basket-analysis/aisles.csv')
    departments = pd.read_csv('./instacart-market-basket-analysis/departments.csv')
    order_products_prior = pd.read_csv('./instacart-market-basket-analysis/order_products__prior.csv')
    order_products_train = pd.read_csv('./instacart-market-basket-analysis/order_products__train.csv')
    orders = pd.read_csv('./instacart-market-basket-analysis/orders.csv')
    products = pd.read_csv('./instacart-market-basket-analysis/products.csv')

    return aisles, departments, order_products_prior, order_products_train, orders, products

def merge_data(order_products_prior, order_products_train, orders, products):
    prior_data = pd.merge(order_products_prior, orders, on='order_id', how='left')
    prior_data = pd.merge(prior_data, products, on='product_id', how='left')

    train_data = pd.merge(order_products_train, orders, on='order_id', how='left')
    train_data = pd.merge(train_data, products, on='product_id', how='left')

    # Keep 'order_hour_of_day' and 'order_dow' columns
    prior_data['order_hour_of_day'] = prior_data['order_hour_of_day'].astype('category')
    prior_data['order_dow'] = prior_data['order_dow'].astype('category')
    
    train_data['order_hour_of_day'] = train_data['order_hour_of_day'].astype('category')
    train_data['order_dow'] = train_data['order_dow'].astype('category')

    return prior_data, train_data


def preprocess_data(data):
    # Fill missing values
    data['days_since_prior_order'].fillna(value=data['days_since_prior_order'].mean(), inplace=True)

    # Encode categorical features
    data['aisle_id'] = data['aisle_id'].astype('category')
    data['department_id'] = data['department_id'].astype('category')

    # Drop unnecessary columns
    data = data.drop(columns=['product_name', 'eval_set'])

    return data

def load_and_preprocess_data(sample_fraction=1.0):
    aisles, departments, order_products_prior, order_products_train, orders, products = load_data()
    prior_data, train_data = merge_data(order_products_prior, order_products_train, orders, products)
    prior_data = preprocess_data(prior_data)
    train_data = preprocess_data(train_data)

    # Sample the data using the provided fraction
    prior_data_sample = prior_data.sample(frac=sample_fraction, random_state=42)
    train_data_sample = train_data.sample(frac=sample_fraction, random_state=42)

    return prior_data_sample , train_data_sample


def create_user_sequences(prior_data, train_data):
    # Concatenate prior_data and train_data
    full_data = pd.concat([prior_data, train_data])

    # Group by user_id and order_number, then aggregate product_id as a list
    user_sequences = full_data.sort_values(['user_id', 'order_number']).groupby(['user_id', 'order_number'])['product_id'].apply(list)

    # Remove the order_number index and convert to dictionary
    user_sequences = user_sequences.reset_index(level='order_number', drop=True).to_dict()

    return user_sequences

def encode_product_ids(user_sequences):
    # Create a label encoder for product IDs
    label_encoder = LabelEncoder()

    # Encode product IDs using the label encoder
    encoded_user_sequences = {user_id: label_encoder.fit_transform(products) for user_id, products in user_sequences.items()}

    # Get the total number of unique product classes
    num_classes = len(label_encoder.classes_)

    return encoded_user_sequences, num_classes