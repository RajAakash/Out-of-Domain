import pandas as pd
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def create_dataframe(data):
    return pd.DataFrame(data, columns=['text', 'intent'])

def prepare_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

def encode_texts(tokenizer, texts, max_length=50):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

def prepare_labels(labels, all_labels=None):
    le = LabelEncoder()
    if all_labels is not None:
        le.fit(all_labels)  # Fit on all possible labels
    else:
        le.fit(labels)
    labels_encoded = le.transform(labels)
    return to_categorical(labels_encoded, num_classes=len(le.classes_)), le

# Load and prepare data
data = load_data('C:/Users/ACER/Downloads/Clinc150TextClassification/clinc150_uci/data_full.json')
train_df = create_dataframe(data['train'])
val_df = create_dataframe(data['val'])
oos_train_df = create_dataframe(data['oos_train'])
oos_val_df = create_dataframe(data['oos_val'])

# Combine all labels for consistent encoding
all_labels = np.concatenate([train_df['intent'], val_df['intent'], oos_train_df['intent'], oos_val_df['intent']])

# Prepare tokenizer
tokenizer = prepare_tokenizer('bert-base-uncased')

# Encode texts
train_encodings = encode_texts(tokenizer, train_df['text'].tolist())
val_encodings = encode_texts(tokenizer, val_df['text'].tolist())
oos_train_encodings = encode_texts(tokenizer, oos_train_df['text'].tolist())
oos_val_encodings = encode_texts(tokenizer, oos_val_df['text'].tolist())

# Prepare labels with consistent encoding
train_labels, label_encoder = prepare_labels(train_df['intent'], all_labels)
val_labels, _ = prepare_labels(val_df['intent'], all_labels)
oos_train_labels, _ = prepare_labels(oos_train_df['intent'], all_labels)
oos_val_labels, _ = prepare_labels(oos_val_df['intent'], all_labels)

# Load BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

# Train the model on in-scope data
model.fit(train_encodings.data, train_labels, batch_size=8, epochs=3, validation_data=(val_encodings.data, val_labels))

# Initial evaluation on in-scope data
print("Initial Evaluation on In-Scope Validation Data:")
model.evaluate(val_encodings.data, val_labels)


# Combined evaluation with out-of-scope data
combined_train_encodings = {key: np.concatenate([train_encodings.data[key], oos_train_encodings.data[key]]) for key in train_encodings.data}
combined_train_labels = np.concatenate([train_labels, oos_train_labels])
combined_val_encodings = {key: np.concatenate([val_encodings.data[key], oos_val_encodings.data[key]]) for key in val_encodings.data}
combined_val_labels = np.concatenate([val_labels, oos_val_labels])

# Re-train the model with combined in-scope and out-of-scope data
print("Re-training with Combined Data:")
model.fit(combined_train_encodings, combined_train_labels, batch_size=8, epochs=3, validation_data=(combined_val_encodings, combined_val_labels))

# Evaluate on combined in-scope and out-of-scope validation data
print("Evaluation on Combined Validation Data:")
model.evaluate(combined_val_encodings, combined_val_labels)


import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Function to convert labels to binary format
def convert_to_binary(classes, oos_index):
    return [1 if class_index == oos_index else 0 for class_index in classes]

# Determine the index for 'out-of-domain' assuming 'oos' is the label for out-of-scope samples
oos_index = np.where(label_encoder.classes_ == 'oos')[0][0]

# Convert true labels for validation set
binary_true_val_labels = convert_to_binary(np.argmax(combined_val_labels, axis=1), oos_index)

# Assuming you've made predictions with the model, and the predictions are logits
logits = model.predict(combined_val_encodings.data).logits
predicted_classes = np.argmax(logits, axis=1)
binary_predicted_val_labels = convert_to_binary(predicted_classes, oos_index)

# Function to apply sigmoid to logits
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Compute metrics
cm = confusion_matrix(binary_true_val_labels, binary_predicted_val_labels)
accuracy = accuracy_score(binary_true_val_labels, binary_predicted_val_labels)
precision = precision_score(binary_true_val_labels, binary_predicted_val_labels)
recall = recall_score(binary_true_val_labels, binary_predicted_val_labels)
roc_auc = roc_auc_score(binary_true_val_labels, sigmoid(logits[:, oos_index]))  # Sigmoid to convert logits to probabilities

print("Confusion Matrix:\n", cm)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("ROC AUC:", roc_auc)


####
# Figure for plotting
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
fpr, tpr, _ = roc_curve(binary_true_val_labels, sigmoid(logits[:, oos_index]))
ax.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.2f})')
ax.plot([0, 1], [0, 1], color='navy', linestyle='--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic')
ax.legend(loc="lower right")

# Annotations for metrics
metrics_text = f"Confusion Matrix:\n{cm}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nROC AUC: {roc_auc:.4f}"
fig.text(0.5, 0.2, metrics_text, ha='center')

# Save figure
plt.savefig('evaluation_metrics.png')
####



# Plot ROC curve
fpr, tpr, _ = roc_curve(binary_true_val_labels, sigmoid(logits[:, oos_index]))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png') 
plt.show()

