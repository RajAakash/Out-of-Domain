import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Step 1: Load the Data
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    texts = []
    labels = []
    for intent_name, samples in data.items():
        for sample in samples:
            texts.append(sample[0])
            labels.append(sample[1])
    return texts, labels

texts, labels = load_data('clinc150_uci/data_full.json')

# Step 2: Preprocess the data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x_data = pad_sequences(sequences, maxlen=50)

encoder = LabelEncoder()
y_data = encoder.fit_transform(labels)
y_data = tf.keras.utils.to_categorical(y_data)

# Split data into training and validation
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

# Step 3: Build the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=50),
    LSTM(64),
    Dense(151, activation='softmax')  
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the model
model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

# Step 5: Evaluate the model
loss, accuracy = model.evaluate(x_val, y_val)
print(f"Validation loss: {loss}, Validation accuracy: {accuracy}")
