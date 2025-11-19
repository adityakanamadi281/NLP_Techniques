#  Data Cleaning 

# Data cleaning involves the process of identifying and correcting errors or inconsistencies in the text data.
# This may includes removing irrelavant charecters, handling misspelled words , and addressing other issues that impact or model performance 


#  Data Cleaning:
#      1. Removing Special Charecters
#      2. Handling missplled words
#      3. Lowercasing


import re 

def clean_text(text):
    text=re.sub(r'[^A-Za-z]+','', text)
    text=re.sub(r'(.)\1+', r'\1', text)
    text=text.lower()
    return text

original_text="Hello @world ! I love programming! NLP is Exciting"
cleaned_text=clean_text(original_text)

print("Original Text : ", original_text)
print("Cleaned text : ", cleaned_text)
