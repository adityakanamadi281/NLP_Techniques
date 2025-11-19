# Named_Entity_Recognition : ----> It is NLP technique that aims to identify and classify entities, such as names of people, organizations.


# NER helps to extract key information from text, enhancing the understanding of documents. Facilitates efficient searching and retrieval of relavant information.
# Essential for systens that answers questions by extracting information from texts.


#  How NER works:
#   - Tokenization: Breaks the text into individual words or tokens.
#   - Part-of-Speech tagging : Assign a part-of-speech tag to each token.
#   - NER : Identify and classify tokens into predefined categories(Entities).
# 


#    Types of Entities:
#       1. Person
#       2. Organization
#       3. Location
#       4. Date 
#       5. Time
#       6. Money
#       7. Percentage



import spacy


nlp = spacy.load("en_core_web_sm")

text= "Sundar Pichai is the CEO of Google and lives in California."

doc= nlp(text)

entities = [(ent.text, ent.label_) for ent in doc.ents]

print("Named Entities : ")
for entity, label in entities:
    print(f"{label}: {entity}")
    