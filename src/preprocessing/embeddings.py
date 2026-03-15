from sentence_transformers import SentenceTransformer
import torch, os, json

class NewsEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        
    def embed_articles(self, articles, max_chars=2000) :
        """Embed a list of articles. Returns tensor (N, dim).

        Args:
            articles (list[dict]): List of article dictionaries.
            max_chars (int): Maximum number of characters to consider for embedding.

        Returns:
            list[list[float]]: List of embedding vectors.
        """
        texts = []
        for article in articles:
            title = article.get('Title', '')
            description = article.get('Description', '')
            keywords = article.get('Keywords', '')
            content = article.get('Content', '')[:max_chars]
            combined_text = f"{title} {description} {keywords} {content}"
            texts.append(combined_text)
            
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True, batch_size=32)
        return embeddings
    
    def save_embeddings(self, date_str, articles, output_dir="data/processed/news_embeddings"):
        """Save embeddings to a JSON file.

        Args:
            embeddings (torch.Tensor): Tensor of shape (N, dim).
            articles (list[dict]): List of article dictionaries.
            output_dir (str): Directory to save the embedding file.
            date_str (str): Date string for file naming.
        """
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        embeddings = self.embed_articles(articles)
        
        torch.save({
            'embeddings': embeddings.cpu(),
            'articles_ids': [a.get('url', str(i)) for i, a in enumerate(articles)],
            'date': date_str
        }, os.path.join(os.path.dirname(output_dir), f"embeddings_{date_str}.pt"))
        
if __name__ == "__main__":
    embedder = NewsEmbedder()
    processed_articles_path = "data/processed_articles_with_sentiment.json"
    
    with open(processed_articles_path, "r") as f:
        articles = json.load(f)
    date_str = "2024-01-01"
    embedder.save_embeddings(date_str, articles)
    