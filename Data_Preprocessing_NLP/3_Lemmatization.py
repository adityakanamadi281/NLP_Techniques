# Lemmatization :  ----> Lemmatization is a text normalisation process that involves reducing the words into their base forms and maintains the meaning of words 


#   Types of Lemmatization
#     1.WordNet Lemmatization
#     2.Rule-Based Lemmatization

import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
print("Downloading WordNet...")
nltk.download("wordnet")


text="Lemmatization Involves reducing words to their form, considering the context and part-of-speech."

words=word_tokenize(text)

lemmatizer=WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

print("Original Words : ", words)
print("Lemmatized Words : ", lemmatized_words)


