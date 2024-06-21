import pandas as pd
import json
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_curve, auc
import seaborn as sns

def load_data(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def create_dataframe(data):
    return pd.DataFrame(data, columns=['text', 'intent'])

def prepare_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

def encode_texts(tokenizer, texts, max_length=50):
    return tokenizer(texts, padding='max_length', truncation=True, max_length=max_length, return_tensors='tf')

def prepare_labels(labels):
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    return to_categorical(labels_encoded), le

# Load and prepare data
data = load_data('/Users/bishalthapa/Desktop/2024/Deep Learning/Project/clinc150_uci/data_full.json')
train_df = create_dataframe(data['train'])
oos_train_df = create_dataframe(data['oos_train'])
val_df = create_dataframe(data['val'])
oos_val_df = create_dataframe(data['oos_val'])

# Combine in-scope and oos training data
combined_train_df = pd.concat([train_df, oos_train_df])
combined_val_df = pd.concat([val_df, oos_val_df])

# Combine all labels for consistent encoding across training and validation
all_labels = np.concatenate([train_df['intent'], val_df['intent'], oos_train_df['intent'], oos_val_df['intent']])
_, label_encoder = prepare_labels(all_labels)

# Encode texts
tokenizer = prepare_tokenizer('bert-base-uncased')
train_encodings = encode_texts(tokenizer, combined_train_df['text'].tolist())
val_encodings = encode_texts(tokenizer, combined_val_df['text'].tolist())

# Prepare labels
train_labels, _ = prepare_labels(label_encoder.transform(combined_train_df['intent']))
val_labels, _ = prepare_labels(label_encoder.transform(combined_val_df['intent']))

# Load BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
# model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])

# Train the model on combined data
model.fit(train_encodings.data, train_labels, batch_size=8, epochs=1, validation_data=(val_encodings.data, val_labels))

# Assuming model prediction
val_predictions = model.predict(val_encodings.data).logits
val_predictions_softmax = tf.nn.softmax(val_predictions).numpy()
val_predicted_labels = np.argmax(val_predictions_softmax, axis=1)

# Convert all in-domain intents to 0 and out-of-scope (oos) to 1 for true labels
binary_true_labels = np.where(combined_val_df['intent'] == 'oos', 1, 0)

# Convert all in-domain intents to 0 and out-of-scope (oos) to 1 for predicted labels
binary_predicted_labels = np.where(label_encoder.inverse_transform(val_predicted_labels) == 'oos', 1, 0)

# Confusion Matrix
cm = confusion_matrix(binary_true_labels, binary_predicted_labels)
print("Confusion Matrix:")
print(cm)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['In-domain', 'OOS'], yticklabels=['In-domain', 'OOS'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.savefig("BERT CM.png")
plt.show()

# Accuracy, Precision, Recall
accuracy = accuracy_score(binary_true_labels, binary_predicted_labels)
precision = precision_score(binary_true_labels, binary_predicted_labels)
recall = recall_score(binary_true_labels, binary_predicted_labels)
print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(binary_true_labels, val_predictions_softmax[:, label_encoder.transform(['oos'])[0]])
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("BERT ROC.png")
plt.show()
