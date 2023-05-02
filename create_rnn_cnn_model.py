import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Concatenate, GlobalMaxPooling1D
from data_preparation import load_and_preprocess_data
from train_test_split import train_test_split_sequences


def create_rnn_cnn_model(sequence_length, num_features, num_classes):
    input_layer = Input(shape=(sequence_length, num_features))
    lstm_layer = LSTM(units=64)(input_layer)

    conv1 = Conv1D(filters=32, kernel_size=2, dilation_rate=1, activation='relu')(input_layer)
    conv2 = Conv1D(filters=32, kernel_size=2, dilation_rate=2, activation='relu')(conv1)
    conv3 = Conv1D(filters=32, kernel_size=2, dilation_rate=4, activation='relu')(conv2)
    conv4 = Conv1D(filters=32, kernel_size=2, dilation_rate=8, activation='relu')(conv3)
    conv5 = Conv1D(filters=32, kernel_size=2, dilation_rate=16, activation='relu')(conv4)
    conv6 = Conv1D(filters=32, kernel_size=2, dilation_rate=32, activation='relu')(conv5)
    pooled_layer = GlobalMaxPooling1D()(conv6)

    concat_layer = Concatenate()([lstm_layer, pooled_layer])

    output_layer = Dense(num_classes, activation='softmax')(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def train_and_evaluate(train_data, train_labels, val_data, val_labels, sequence_length, num_features, num_classes):
    # Prepare the data
    x_train = np.array([sequence for user_id, sequence in train_data.items()])
    y_train = np.array([label for user_id, label in train_data.items()])
    x_val = np.array([sequence for user_id, sequence in val_data.items()])
    y_val = np.array([label for user_id, label in val_data.items()])

    # Create the RNN/CNN model
    model = create_rnn_cnn_model(sequence_length, num_features, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

    # Evaluate the model's performance
    performance_metrics = model.evaluate(x_val, y_val)

    return performance_metrics

def prepare_data_for_model(train_data, val_data):
    # Extract data_sequences and data_labels from train_data
    train_sequences = [user_data['sequences'] for user_data in train_data.values()]
    train_labels = [user_data['labels'] for user_data in train_data.values()]

    # Extract data_sequences and data_labels from val_data
    val_sequences = [user_data['sequences'] for user_data in val_data.values()]
    val_labels = [user_data['labels'] for user_data in val_data.values()]

    # Convert the data to numpy arrays
    train_sequences = np.array(train_sequences)
    train_labels = np.array(train_labels)
    val_sequences = np.array(val_sequences)
    val_labels = np.array(val_labels)

    return train_sequences, train_labels, val_sequences, val_labels





# Load and preprocess the data
train_data, val_data = load_and_preprocess_data(sample_fraction=1.0)

# Prepare the data for the RNN/CNN model
train_sequences, train_labels, val_sequences, val_labels = prepare_data_for_model(train_data, val_data)

# Set sequence_length, num_features, and num_classes based on your data
sequence_length = len(train_sequences[0])
num_features = len(train_sequences[1])
num_classes = len(train_labels[0])

# Train and evaluate the RNN/CNN model
performance_metrics = train_and_evaluate(train_sequences, train_labels, val_sequences, val_labels, sequence_length, num_features, num_classes)

print("Performance metrics:", performance_metrics)
