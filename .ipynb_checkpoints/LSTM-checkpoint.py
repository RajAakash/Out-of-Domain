import pandas as pd
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load data from JSON file
with open('clinc150_uci/data_full.json', 'r') as file:
    data = json.load(file)

# Convert lists to DataFrames
train_df = pd.DataFrame(data['train'], columns=['text', 'intent'])
val_df = pd.DataFrame(data['val'], columns=['text', 'intent'])
test_df = pd.DataFrame(data['test'], columns=['text', 'intent'])
oos_train_df=pd.DataFrame(data['oos_train'],columns=['text','intent'])
oos_test_df=pd.DataFrame(data['oos_test'],columns=['text','intent'])
oos_val_df=pd.DataFrame(data['oos_val'],columns=['text','intent'])

# Prepare the tokenizer for LSTM
lstm_tokenizer = Tokenizer(num_words=10000)  # Adjust `num_words` as needed
lstm_tokenizer.fit_on_texts(train_df['text'])  # Only fit on training data

# Preprocess the data for LSTM
max_length = 100  # or some other value that makes sense for your data
train_sequences = lstm_tokenizer.texts_to_sequences(train_df['text'])
train_sequences_padded = pad_sequences(train_sequences, maxlen=max_length)

val_sequences = lstm_tokenizer.texts_to_sequences(val_df['text'])
val_sequences_padded = pad_sequences(val_sequences, maxlen=max_length)

# Convert labels to categorical values
label_index = {label: idx for idx, label in enumerate(set(train_df['intent']))}
train_labels = to_categorical(np.array(train_df['intent'].map(label_index)))
val_labels = to_categorical(np.array(val_df['intent'].map(label_index)))

# Build and compile the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=10000, output_dim=128, input_length=max_length))
lstm_model.add(LSTM(units=64))
lstm_model.add(Dense(len(label_index), activation='softmax'))
lstm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the LSTM model
lstm_model.fit(train_sequences_padded, train_labels, batch_size=128, epochs=10, validation_data=(val_sequences_padded, val_labels))

# Prepare the tokenizer for Transformer (BERT)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 64  # BERT's maximum sequence length is 512, choose a suitable length for your data

# Function to preprocess texts for BERT
def bert_preprocess(df, tokenizer):
    input_ids, attention_masks = [], []
    for text in df['text']:
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return tf.concat(input_ids, 0), tf.concat(attention_masks, 0)

# Preprocess the data for BERT
train_inputs, train_masks = bert_preprocess(train_df, bert_tokenizer)
val_inputs, val_masks = bert_preprocess(val_df, bert_tokenizer)

# Load pre-trained BERT model for sequence classification
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_index))

# Compile the BERT model
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.CategoricalAccuracy('accuracy')

bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train the BERT model
bert_history = bert_model.fit(
    [train_inputs, train_masks],
    train_labels,
    batch_size=32,
    epochs=3,
    validation_data=([val_inputs, val_masks], val_labels)
)


# Additional preprocessing for the out-of-sample data
oos_train_sequences = lstm_tokenizer.texts_to_sequences(oos_train_df['text'])
oos_train_sequences_padded = pad_sequences(oos_train_sequences, maxlen=max_length)

oos_val_sequences = lstm_tokenizer.texts_to_sequences(oos_val_df['text'])
oos_val_sequences_padded = pad_sequences(oos_val_sequences, maxlen=max_length)

oos_test_sequences = lstm_tokenizer.texts_to_sequences(oos_test_df['text'])
oos_test_sequences_padded = pad_sequences(oos_test_sequences, maxlen=max_length)

# Create an additional 'oos' or 'unknown' label for out-of-sample data
oos_label = 'oos'
if oos_label not in label_index:
    label_index[oos_label] = len(label_index)

oos_train_labels = to_categorical([label_index[oos_label]] * oos_train_sequences_padded.shape[0])
oos_val_labels = to_categorical([label_index[oos_label]] * oos_val_sequences_padded.shape[0])
oos_test_labels = to_categorical([label_index[oos_label]] * oos_test_sequences_padded.shape[0])

# Combine in-domain and out-of-sample training data
combined_train_sequences_padded = np.concatenate([train_sequences_padded, oos_train_sequences_padded])
combined_train_labels = np.concatenate([train_labels, oos_train_labels])

# Retrain the LSTM model with combined in-domain and OOS training data
lstm_model.fit(combined_train_sequences_padded, combined_train_labels, batch_size=128, epochs=10, validation_data=(val_sequences_padded, val_labels))

# Evaluate the LSTM model on the combined validation data, including OOS samples
combined_val_sequences_padded = np.concatenate([val_sequences_padded, oos_val_sequences_padded])
combined_val_labels = np.concatenate([val_labels, oos_val_labels])
lstm_model.evaluate(combined_val_sequences_padded, combined_val_labels)

# Finally, evaluate the LSTM model on the OOS test set
lstm_model.evaluate(oos_test_sequences_padded, oos_test_labels)