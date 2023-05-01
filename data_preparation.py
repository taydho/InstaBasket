import pandas as pd

def load_and_preprocess_data():
    aisles = pd.read_csv('aisles.csv')
    departments = pd.read_csv('departments.csv')
    order_products_prior = pd.read_csv('order_products__prior.csv')
    order_products_train = pd.read_csv('order_products__train.csv')
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')

    products_extended = pd.merge(products, aisles, on='aisle_id')
    products_extended = pd.merge(products_extended, departments, on='department_id')

    prior_data = pd.merge(order_products_prior, orders, on='order_id')
    prior_data = pd.merge(prior_data, products_extended, on='product_id')

    train_data = pd.merge(order_products_train, orders, on='order_id')
    train_data = pd.merge(train_data, products_extended, on='product_id')

    return prior_data, train_data
