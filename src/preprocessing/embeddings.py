from sentence_transformers import SentenceTransformer
from collections import defaultdict
import torch, os, json

class NewsEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_articles(self, articles, max_chars=2000, output_dir="embeddings") :
        """Embed a list of articles. Returns tensor (N, dim).

        Args:
            articles (list[dict]): List of article dictionaries.
            max_chars (int): Maximum number of characters to consider for embedding.

        Returns:
            list[list[float]]: List of embedding vectors.
        """

        by_date = defaultdict(list)
        for article in articles:
            date = article.get('Date', 'unknown')
            by_date[date].append(article)
        
        print(f"Found {len(by_date)} unique dates across {len(articles)} articles")
        
        daily_data = {}
        for date in sorted(by_date.keys()):
            day_articles = by_date[date]
            texts = []
            for article in articles:
                title = article.get('Title', '')
                description = article.get('Description', '')
                content = article.get('Content', '')[:max_chars]
                combined_text = f"{title} {description} {content}"
                texts.append(combined_text)

            embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_tensor=True, batch_size=32)
            torch.save({
            'embeddings': embeddings,           # (N_articles_that_day, 384)
            'articles': day_articles,           # list of article dicts
            'date': date,
            }, os.path.join(output_dir, f"{date}.pt"))
        
            daily_data[date] = {'embeddings': embeddings, 'articles': day_articles}
            print(f"  {date}: {len(day_articles)} articles → embeddings {embeddings.shape}")
        
        return daily_data

    def save_embeddings(self, date_str, articles, output_dir="news_embeddings"):
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
    processed_articles_path = "processed_articles_with_sentiment_3rdApril.json"

    with open(processed_articles_path, "r") as f:
        articles = json.load(f)
    embedder.embed_articles(articles)
