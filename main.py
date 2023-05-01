from data_preparation import load_and_preprocess_data, create_user_sequences, encode_product_ids
from create_rnn_cnn_model import train_and_evaluate
from train_test_split import train_test_split_sequences

def main():
    # Load and preprocess the data
    prior_data, train_data = load_and_preprocess_data()

    # Create sequences of ordered products for each user
    user_sequences = create_user_sequences(prior_data, train_data)

    # Encode product IDs
    encoded_user_sequences, num_classes = encode_product_ids(user_sequences)

    # Split the data into training and validation sets
    train_data, val_data = train_test_split_sequences(encoded_user_sequences, test_size=0.1, random_state=42)

    # Set the input dimensions for the RNN/CNN model
    sequence_length = train_data.shape[1]
    num_features = train_data.shape[2]

    # Train and evaluate the RNN/CNN model
    performance_metrics = train_and_evaluate(train_data, val_data, sequence_length, num_features, num_classes)

    print(performance_metrics)

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    duration = end_time - start_time
    print(f'Training completed in {duration // 60:.0f}m {duration % 60:.0f}s')
