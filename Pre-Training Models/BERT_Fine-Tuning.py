from transformers import BertForSequenceClassification, BertTokenizer, AdamW
from torch.utils.data import DataLoader

# Load pre-trained BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare dataset and dataloader (assuming 'texts' and 'labels' are defined)
dataset = [(tokenizer(text, padding='max_length', truncation=True, max_length=128), label) for text, label in zip(texts, labels)]
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Fine-tuning loop
optimizer = AdamW(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in dataloader:
        inputs = {k: v.to(model.device) for k, v in batch[0].items()}
        labels = batch[1].to(model.device)
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save fine-tuned model
model.save_pretrained('./fine_tuned_bert_classifier')