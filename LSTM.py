import pandas as pd
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def create_dataframe(data, columns):
    return pd.DataFrame(data, columns=columns)

def prepare_tokenizer(texts, num_words=10000):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    return tokenizer

def process_sequences(tokenizer, texts, max_length):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length)

def encode_labels(labels, label_index):
    label_indices = [label_index[label] for label in labels]
    return to_categorical(label_indices, num_classes=len(label_index))

def build_lstm_model(input_dim, output_dim, input_length, num_classes):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=output_dim, input_length=input_length),
        LSTM(units=64),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load and prepare data
data = load_data('C:/Users/ACER/Downloads/Clinc150TextClassification/clinc150_uci/data_full.json')
train_df = create_dataframe(data['train'], ['text', 'intent'])
val_df = create_dataframe(data['val'], ['text', 'intent'])
test_df = create_dataframe(data['test'], ['text', 'intent'])
oos_train_df = create_dataframe(data['oos_train'], ['text', 'intent'])
oos_val_df = create_dataframe(data['oos_val'], ['text', 'intent'])
oos_test_df = create_dataframe(data['oos_test'], ['text', 'intent'])

# Prepare tokenizer and sequences
tokenizer = prepare_tokenizer(train_df['text'])
train_sequences_padded = process_sequences(tokenizer, train_df['text'], max_length=100)
val_sequences_padded = process_sequences(tokenizer, val_df['text'], max_length=100)

# Label encoding
all_labels = set(train_df['intent']).union(val_df['intent'], test_df['intent'], ['oos'])
label_index = {label: idx for idx, label in enumerate(all_labels)}
train_labels = encode_labels(train_df['intent'], label_index)
val_labels = encode_labels(val_df['intent'], label_index)

# Build and train LSTM model
lstm_model = build_lstm_model(input_dim=10000, output_dim=128, input_length=100, num_classes=len(label_index))
lstm_model.fit(train_sequences_padded, train_labels, batch_size=128, epochs=10, validation_data=(val_sequences_padded, val_labels))

# Save tokenizer and model for later use
tokenizer.save('tokenizer.json')
lstm_model.save('lstm_model.h5')

# Evaluate on combined in-domain and out-of-sample validation data
oos_val_sequences_padded = process_sequences(tokenizer, oos_val_df['text'], max_length=100)
oos_val_labels = encode_labels(['oos'] * len(oos_val_df), label_index)
combined_val_sequences_padded = np.concatenate([val_sequences_padded, oos_val_sequences_padded])
combined_val_labels = np.concatenate([val_labels, oos_val_labels])
lstm_model.evaluate(combined_val_sequences_padded, combined_val_labels)
