import pandas as pd
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

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

def convert_to_binary(classes, oos_index):
    return [0 if class_index != oos_index else 1 for class_index in classes]

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

# Generate predictions for combined validation dataset
combined_val_predictions = model.predict(oos_val_encodings.data)
combined_val_predictions_classes = np.argmax(combined_val_predictions.logits, axis=1)

# Assuming `oos_index` is the index of 'oos' in label_encoder.classes_
oos_index = np.where(label_encoder.classes_ == 'oos')[0][0]

# Convert true labels and predicted labels for validation
binary_true_val_labels = convert_to_binary(np.argmax(val_labels, axis=1), oos_index)
binary_predicted_val_labels = convert_to_binary(combined_val_predictions_classes, oos_index)

# Create a confusion matrix plot
plt.figure(figsize=(10, 8))
cm = confusion_matrix(binary_true_val_labels, binary_predicted_val_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['In-domain', 'Out-of-domain'], yticklabels=['In-domain', 'Out-of-domain'])
plt.title('Confusion Matrix for Classification')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')  # Save as PNG
plt.close()

# Create a ROC curve plot
plt.figure(figsize=(10, 8))
fpr, tpr, thresholds = roc_curve(binary_true_val_labels, combined_val_predictions.logits[:, oos_index])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')  # Save as PNG
plt.close()

# Save classification report as a text file
with open('classification_report.txt', 'w') as file:
    report = classification_report(binary_true_val_labels, binary_predicted_val_labels, target_names=['In-domain', 'Out-of-domain'])
    file.write(report)
