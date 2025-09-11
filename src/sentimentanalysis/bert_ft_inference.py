import torch
from transformers import BertForSequenceClassification, BertTokenizer


def predict_sentiment(text, model, tokenizer, device="cpu"):
    model.eval()
    model.to(device)
    
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        confidence, predicted_class_id = torch.max(probabilities, dim=-1)
        
        confidence = confidence.item()
        predicted_class_id = predicted_class_id.item()
        probabilities = probabilities.cpu().numpy()[0]
    
    return confidence, probabilities

model_path = "./imdb_sentiment_full"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)


confidence, probs = predict_sentiment(
    "The director did a marvelous job with this film!", 
    model,
    tokenizer
)
print(f"Result: ({confidence:.2%} confident)")