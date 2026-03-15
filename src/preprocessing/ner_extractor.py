import csv
from typing import Any
import spacy 
from rapidfuzz import process, fuzz
import json

class NERLinker:
    def __init__(self, companies_path="config/nifty50_tickers.json"):
        self.nlp = spacy.load("en_core_web_sm")
        with open(companies_path) as f:
            self.companies = json.load(f)
            
        self.aliases = {}
    
        for company in self.companies['tickers']:
            symbol = company['symbol']
            name = company['name']
            clean_ticker = symbol.replace(".NS", "")
            
            self.aliases[name.lower()] = company['symbol']
            self.aliases[name.title()] = company['symbol']
            self.aliases[name.upper()] = company['symbol']
            self.aliases[clean_ticker.lower()] = company['symbol']
            
            for word in name.split():
                if len(word) > 3:
                    self.aliases[word.lower()] = company['symbol']
                    
        self.aliases.update({
            'sbi': 'SBIN.NS', 
        })
        
    def extract_entities(self, text) -> tuple[list[str], list[str]]:
        """Extract ORG entities and link to NIFTY 50 tickers

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        doc = self.nlp(text)
        matched_entities = set()
        raw_entities = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:
                raw_entities.append(ent.text)
                
                if ent.text.lower() in self.aliases:
                    matched_entities.add(self.aliases[ent.text.lower()])
                else:
                    match = process.extractOne(ent.text.lower(), self.aliases.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=80)
                    if match:
                        matched_entities.add(self.aliases[match[0]])
        
        return list(matched_entities), raw_entities
    
    def process_articles(self, articles) -> list[Any]:
        
        for article in articles[0:10]:
            text = f"{article['Title']} {article['Content']}"
            tickers, entities = self.extract_entities(text)
            article['matched_entities'] = tickers
            article['raw_entities'] = entities
            
        return articles


if __name__ == "__main__":
    linker = NERLinker()
    news_data_path = "data/IN-FINews Dataset.csv"
    # aliases_path = "data/aliases.json"
    # aliases = linker.aliases
    # save_aliases_path = "data/processed_aliases.json"
  
    # with open(save_aliases_path, 'w') as f:        
    #     json.dump(aliases, f, indent=4)
    
    
    print("Loading news data...")
    articles = []
    with open(news_data_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            articles.append(row)
    print("Processing articles...")
    processed_articles = linker.process_articles(articles)
    save_path = "data/processed_articles3.json"
    with open(save_path, 'w') as f:
        json.dump(processed_articles, f, indent=4)
    print(f"Processed articles saved to {save_path}")
    
        

    