import pandas as pd

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

    # Keep 'order_hour_of_day', 'order_dow', 'aisle_id', and 'department_id' columns
    prior_data['order_hour_of_day'] = prior_data['order_hour_of_day'].astype('category')
    prior_data['order_dow'] = prior_data['order_dow'].astype('category')
    prior_data['aisle_id'] = prior_data['aisle_id'].astype('category')
    prior_data['department_id'] = prior_data['department_id'].astype('category')
    
    train_data['order_hour_of_day'] = train_data['order_hour_of_day'].astype('category')
    train_data['order_dow'] = train_data['order_dow'].astype('category')
    train_data['aisle_id'] = train_data['aisle_id'].astype('category')
    train_data['department_id'] = train_data['department_id'].astype('category')

    return prior_data, train_data



def preprocess_data(data):
    # Fill missing values
    data['days_since_prior_order'].fillna(value=data['days_since_prior_order'].mean(), inplace=True)

    # Encode categorical features
    data['aisle_id'] = data['aisle_id'].astype('category')
    data['department_id'] = data['department_id'].astype('category')

    return data



def load_and_preprocess_data(sample_fraction=1.0):
    aisles, departments, order_products_prior, order_products_train, orders, products = load_data()
    
    # Merge products, aisles, and departments
    products = products.merge(aisles, on='aisle_id').merge(departments, on='department_id')
    
    prior_data, train_data = merge_data(order_products_prior, order_products_train, orders, products)
    prior_data = preprocess_data(prior_data)
    train_data = preprocess_data(train_data)

    # Sample the data using the provided fraction
    prior_data_sample = prior_data.sample(frac=sample_fraction, random_state=42)
    train_data_sample = train_data.sample(frac=sample_fraction, random_state=42)

    return prior_data_sample, train_data_sample, products, aisles, departments
