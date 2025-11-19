# Stopword Removal : It is text preprocessing step where common words that do not contribute significant meaning to the text are eliminated.


#  Working:
#    1. Tokenization
#    2. Stopwords
#    3. Removal


import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


text = "Stopword removal is an important step in natural language processing. It helps improve text analysis."

words = word_tokenize(text)

stop_words = set(stopwords.words("english"))

filtered_words = [word for word in words if word.lower() not in stop_words]

print("original Words : ", words)
print("Filtered words (without stopwords): ", filtered_words)
