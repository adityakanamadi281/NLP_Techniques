import numpy as np
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense , Activation, Bidirectional
from tensorflow.keras.optimizers import Adam 
import matplotlib.pyplot as plt
import pickle
import heapq 
import requests


url="https://github.com/simranjeet97/75DayHard_GenAI_LLM_Challenge/blob/main/NextWordPrediction_DeepLearning/1661-0.txt"

response = requests.get(url)


if response.status_code==200:
    text=response.text.lower()
    print("corpus length:", len(text))
else:
    print(f"Error fetching file. Status code: {response.status_code}")



tokenizer= RegexpTokenizer(r"\w+")

words = tokenizer.tokenize(text)

unique_words=np.unique(words)
unique_word_index=dict((c,i) for i,c in enumerate(unique_words))




WORD_LENGTH = 5
prev_words = []
next_words=[]

for i in range(len(words)-WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i+ WORD_LENGTH])

print("Prev_Words :",prev_words[0])
print("Next_Words :",next_words[0])



x = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=bool)
y = np.zeros((len(next_words), len(unique_words)), dtype=bool)




for i,each_words in enumerate(prev_words):
    for j,each_word in enumerate(each_words):
        x[i,j,unique_word_index[each_word]]=1
    y[i,unique_word_index[next_words[i]]]=1


print(unique_word_index['new'])

# LSTM model 

model=Sequential()
model.add(LSTM(128,input_shape=(WORD_LENGTH,len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation("softmax"))

optimizer=Adam(learning_rate=0.01)

model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy'])

history=model.fit(x,y,validation_split=0.05, batch_size=128, epochs=5, shuffle=True).history




model.save('next_word_model.h5')
pickle.dump(history, open("history.p", "wb"))
model = load_model('next_word_model.h5')
history = pickle.load(open("history.p", "rb"))




plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')




plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')



def prepare_input(text):
    x = np.zeros((1, WORD_LENGTH, len(unique_words)))
    for t, word in enumerate(text.split()):
        print(word)
        x[0, t, unique_word_index[word]] = 1
    return x

prepare_input("It is not a lack".lower()) 



def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)





def predict_completions_with_probabilities(text, n=3):
    if text == "":
        return [("0", 0.0)]  
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    probabilities = [preds[idx] for idx in next_indices]
    predicted_words = [unique_words[idx] for idx in next_indices]
    return list(zip(predicted_words, probabilities))



q = "Your life will never be there in the same situation again"
print("Correct sentence:", q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence:", seq)



predictions_with_probabilities = predict_completions_with_probabilities(seq, 5)

# Display predictions with probabilities
for word, probability in predictions_with_probabilities:
    print(f"Word: {word}, Probability: {probability * 100}")





# Define the bidirectional LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(128), input_shape=(WORD_LENGTH, len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))



optimizer = Adam(learning_rate=0.01)

# Compile the model with the Adam optimizer
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model with the Adam optimizer
history_2 = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=5, shuffle=True).history




def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return heapq.nlargest(top_n, range(len(preds)), preds.take)



def predict_completions_with_probabilities(text, n=3):
    if text == "":
        return [("0", 0.0)]  # Return a tuple containing a placeholder and probability 0.0
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    probabilities = [preds[idx] for idx in next_indices]
    predicted_words = [unique_words[idx] for idx in next_indices]
    return list(zip(predicted_words, probabilities))



q = "Your life will never be there in the same situation again"
print("Correct sentence:", q)
seq = " ".join(tokenizer.tokenize(q.lower())[0:5])
print("Sequence:", seq)

# Get predictions with probabilities for each word in the sequence
predictions_with_probabilities = predict_completions_with_probabilities(seq, 5)

# Display predictions with probabilities
for word, probability in predictions_with_probabilities:
    print(f"Word: {word}, Probability: {probability * 100}")


print("Unidirectional LSTM - Validation Accuracy:", history['val_accuracy'][-1]*100)
print("Bidirectional LSTM - Validation Accuracy:", history_2['val_accuracy'][-1]*100)



