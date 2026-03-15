from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import json

class FinBERTSentiment:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.tokenizer = AutoTokenizer.from_pretrained("Vansh180/FinBERT-India-v1")
        self.model = AutoModelForSequenceClassification.from_pretrained("Vansh180/FinBERT-India-v1", num_labels=3).to(device)
        self.model.to(device).eval()
        self.device = device
        self.labels = ["positive", "negative", "neutral"]
        
    def predict(self, text, batch_size=16):
        """Return list of dicts with sentiment label and scores.

        Args:
            text (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 16.

        Returns:
            _type_: _description_
        """
        results = []
        for i in range(0, len(text), batch_size):
            batch = text[i:i+batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        
            with torch.no_grad():
                outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            
            for p in probs:
                label_idx = np.argmax(p)
                results.append({
                    "label": self.labels[label_idx],
                    "positive": float(p[0]),
                    "negative": float(p[1]),
                    "neutral": float(p[2]),
                    "score": float(p[0] - p[1])
                })        
            return results
        
if __name__ == "__main__":
    finbert = FinBERTSentiment()
    processed_articles_path = "processed_articles4.json"
    
    with open(processed_articles_path, "r") as f:
        articles = json.load(f)
    texts = [f"{a['Title']} {a['Content']}" for a in articles[0:10]]
    sentiments = finbert.predict(texts)
    for article, sentiment in zip(articles[0:10], sentiments):
        article["sentiment"] = sentiment
    with open("processed_articles_with_sentiment.json", "w") as f:
        json.dump(articles, f, indent=4)
