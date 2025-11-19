# Parts of Speech Tagging  : ---->  It is the process of assigning grammatical categories to words in a sentence based on their syntactic and semantic roles.

#         It helps to understand the grammatical structure of a sentence.


#  POS tagging involves analyzing the context of each word in a sentence and assigning the appropriate part-of-speech tag.


#    Types of POS Tagging
#       1. Noun(NN):      ---> Represents a person, place, thing, or idea
#       2. Verb(VB):      ---> Represents an action or state of being
#       3. Adjective(JJ): ---> Describes a noun or pronoun. 
#       4. Adverb(RB) :   ---> Describes a verb, adjective, or other adverb
#       4. Pronoun(PRP):  ---> Replaces a noun in a sentence.
#       5. Preposition(IN)---> Release a noun to another word
#       6. Conjuction(CC):---> Connects words or groups of words.
#       7. Interjection(UH)--> Expresses strong emotion.
#       8. Determiner(DT):---> Specifies a noun as definite or indefinite.
#       9. Practice(RP) : ---> Small words that don't fit into other categories.



import nltk
from nltk.tokenize import word_tokenize
# nltk.download("punkt")
# nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")


text="part-of-sppech tagging helps in understanding the grammatical structure of a sentence."

words=word_tokenize(text)

pos_tags=nltk.pos_tag(words)


print("Original Words : ", words)
print("POS tags : ", pos_tags)



