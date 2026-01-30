# Text Classification  with HuggingFace

from transformers import pipeline 

classifier = pipeline("sentiment-analysis") 
text = "I really like to work with transfomers library and LLMs use cases." 

result = classifier(text) 
print(f"Text : {result}") 
print(f"Predicted Sentiment: {result[0]['label']} with confidence {result[0]['score']}")



# Text Classification _ NER

from transformers import pipeline

ner_pipeline = pipeline("ner")

sample_text = "Mathematics is easy to learn and practice , and HuggingFace is fantastic platform to do NLP tasks."

ner_results = ner_pipeline(sample_text)

print(f"sample text : {sample_text}")
print("NER Results : ")

for entity in ner_results:
  print(f"Entity : {entity["word"]}, label : {entity["entity"]}, score : {entity["score"]}")




# Table Question Answering  with HF

from transformers import pipeline 
import pandas as pd

data = {"Actors" : ["Rajakumar","Shankarnag","Ravichandran"], "Number of movies": ["200","85","90"]}

table = pd.DataFrame(data)

question = "How many movies rajakumar have"

tqa = pipeline(task="table-question-answering", model="google/tapas-large-finetuned-wtq")

print(tqa(table=table, query=question)["cells"][0]) 



# Normal Question Answering  with HF 

from transformers import pipeline

question_answer = pipeline("question-answering")

context = "Aditya Kanamadi is an AI and data scientist, lives in Bangalore since 2022"

question = "How long been aditya kanamadi living in bangalore?"

qa_results = question_answer(question=question, context=context)

print(f"Question : {question}")
print(f"Answer : {qa_results["answer"]}")
print(f"Score : {qa_results["score"]}")



# Language Translation  with HF 

from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-fr")

text_to_translate = "How are you?"

tr_results = translator(text_to_translate)

print(f"Text : {text_to_translate}")
print(f'Translated text: {tr_results[0]["translation_text"]}')



