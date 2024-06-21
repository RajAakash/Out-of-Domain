import json

# Load the data from a file
with open('C:/Users/ACER/Downloads/Clinc150TextClassification/clinc150_uci/data_full.json', 'r') as file:
    data = json.load(file)
    print(data)

# Print out the number of samples in each subset
for subset in data:
    print(f"{subset}: {len(data[subset])} samples")

def calculate_average_length(texts):
    if texts and isinstance(texts[0], list):
        # If the data is already tokenized into lists of words
        return mean(len(text) for text in texts)
    elif texts and isinstance(texts[0], str):
        # If the data is raw strings
        return mean(len(text.split()) for text in texts)
    else:
        return 0  # In case the texts are empty or not a list of strings/lists

avg_length_train = calculate_average_length(data['train'])
print(f"Average length of training texts: {avg_length_train} words")

from collections import Counter
import matplotlib.pyplot as plt

# Assuming 'data' is your dataset loaded from JSON

# Combine labels from all subsets
all_labels = []
for subset in ['oos_val', 'val', 'train', 'oos_test', 'test', 'oos_train']:
    all_labels.extend([entry[1] for entry in data[subset]])

# Count the frequency of each label
label_counts = Counter(all_labels)

# Display the frequency of each label
print(label_counts)

# Bar chart visualization of label frequencies
plt.figure(figsize=(23, 8))
plt.bar(label_counts.keys(), label_counts.values())
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Frequency of Each Label Across All Data Subsets')
plt.xticks(rotation=90)  # Rotate labels to prevent overlap
plt.tight_layout()  # Adjust layout to show data
plt.show()

import re
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Preprocess the texts: lowercasing, removing special characters, and stemming
def preprocess(text):
    # Join the tokens back into a string if it's a list
    if isinstance(text, list):
        text = ' '.join(text)
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)  # Remove all non-word characters
    words = text.split()
    words = [stemmer.stem(word) for word in words]  # Stemming
    return ' '.join(words)

# Example usage with your data structure
preprocessed_data = {
    key: [preprocess(item[0]) for item in value] 
    for key, value in data.items()
}

from sklearn.feature_extraction.text import CountVectorizer

# Use CountVectorizer to find frequency of words in the training data
vectorizer = CountVectorizer(max_features=1000)  # considering top 1000 words
X_train = vectorizer.fit_transform(preprocessed_data['train'])

# Sum up the counts of each vocabulary word
word_counts = X_train.sum(axis=0)

# Map from vocabulary word index to actual word
words_freq = [(word, word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)

# Display the top 10 words
print(words_freq[:10])