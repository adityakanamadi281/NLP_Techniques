# Stemming :  Stemming reduces the word to it's base form 

#    types of stemming algorithms 
#         1. The Porter Stemmer
#         2. Snowball Stemmer
#          



# 1. The Porter Stemmer

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

words = ["running", "flies", "happily", "better", "playing"]
stemmed_words=[stemmer.stem(word) for word in words]

print("original words : ", words)
print("Porter Stemmed words : ", stemmed_words)




# 2. Snowball Stemmer

from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")

snowball_stemmed_words=[snowball_stemmer.stem(word) for word in words]

print("Original Words : ", words)
print("Snowball Stemmed Words : ", snowball_stemmed_words)

