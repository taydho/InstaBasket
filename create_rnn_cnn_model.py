import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Conv1D, Dense, Concatenate
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

    concat_layer = Concatenate()([lstm_layer, conv6])

    output_layer = Dense(num_classes, activation='softmax')(concat_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def train_and_evaluate(train_data, train_labels, val_data, val_labels, sequence_length, num_features, num_classes):
    # Create the RNN/CNN model
    model = create_rnn_cnn_model(sequence_length, num_features, num_classes)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_data=(val_data, val_labels))

    # Evaluate the model's performance
    performance_metrics = model.evaluate(val_data, val_labels)

    return performance_metrics

def prepare_data_for_model(train_data, val_data):
    # Extract data_sequences and data_labels from train_data
    train_sequences = [sequence for user_id, sequence in train_data.items()]
    train_labels = [label for user_id, label in train_data.items()]

    # Extract data_sequences and data_labels from val_data
    val_sequences = [sequence for user_id, sequence in val_data.items()]
    val_labels = [label for user_id, label in val_data.items()]

    # Split the data into training and validation sets
    train_sequences, train_labels, val_sequences, val_labels = train_test_split_sequences(train_sequences, train_labels)

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
