import numpy as np

def train_test_split_sequences(data_sequences, data_labels, test_size=0.1, random_state=None):
    if random_state:
        np.random.seed(random_state)

    num_samples = len(data_sequences)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_idx = int(num_samples * (1 - test_size))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_sequences = [data_sequences[i] for i in train_indices]
    train_labels = [data_labels[i] for i in train_indices]

    val_sequences = [data_sequences[i] for i in val_indices]
    val_labels = [data_labels[i] for i in val_indices]

    return train_sequences, train_labels, val_sequences, val_labels

