# -*- coding: utf-8 -*-

#NOTES
#categorical data poses an extra challenge 

#convert all files to text based files? 
#USE BERT?

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, concatenate, Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, TFBertModel, TFBertForSequenceClassification, TextDataset, TrainingArguments, Trainer
# Assuming X_uncleaned and X_cleaned are your datasets
# Preprocessing: Standardize numerical data and Integer encode categorical data

# Define paths
uncleaned_files_path = '/path/to/uncleaned_text_files'
cleaned_files_path = '/path/to/cleaned_text_files'

# List all files
#uncleaned_files = os.listdir(uncleaned_files_path)
#cleaned_files = os.listdir(cleaned_files_path)

# Read files into lists
def read_files_into_list(files, path):
   data_list = []
   for file in files:
       with open((path, file), 'r', encoding='utf-8') as f:
           data_list.append(f.read())
   return data_list

#TODO: CREATE LISTS OF CLEANED AND UNCLEANED FILES FROM OLD PRE PROCESSED FILES FROM SIMIO AND TRANSFORMED FILES FROM SIMIO
#WILL HAVE PROBLEM WITH CASS DATA, MAY NEED WORKFLOW WITHOUT CASS DATA OR ACCESS TO THE CASS API 

# 1. Data Preparation
# Assuming you've read your uncleaned and cleaned data into lists:
#
#    
#ERROR HERE NEED THE FILES 
#
#    
uncleaned_texts = read_files_into_list(uncleaned_files, uncleaned_files_path)
cleaned_texts = read_files_into_list(cleaned_files, cleaned_files_path)
#
#    
#    


# Tokenization using BERT's tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
uncleaned_encodings = tokenizer(uncleaned_texts, truncation=True, padding='max_length', max_length=254, return_tensors='tf')
cleaned_encodings = tokenizer(cleaned_texts, truncation=True, padding='max_length', max_length=254, return_tensors='tf')

# Train-test split
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(uncleaned_encodings['input_ids'], cleaned_encodings['input_ids'], test_size=0.2)

# 2. Model Building
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

input_ids = tf.keras.layers.Input(shape=(254,), dtype=tf.int32)
embeddings = bert_model(input_ids).last_hidden_state

# Use LSTM for the sequence-to-sequence task
lstm_out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(254, return_sequences=True))(embeddings)
output = tf.keras.layers.Dense(len(tokenizer.vocab), activation="softmax")(lstm_out) # Predicting word in vocab

model = tf.keras.Model(inputs=input_ids, outputs=output)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# 3. Model Training
model.fit(train_inputs, train_outputs, validation_split=0.1, epochs=3)

# 4. Model Evaluation (optional)
loss = model.evaluate(test_inputs, test_outputs)
print(f"Test Loss: {loss}")

# For predictions
predicted_output = model.predict(test_inputs)
predicted_tokens = tf.argmax(predicted_output, axis=-1)
predicted_texts = tokenizer.batch_decode(predicted_tokens)
