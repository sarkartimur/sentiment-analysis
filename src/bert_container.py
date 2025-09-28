import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer, AutoModel
from constants import BERT_MODEL, BERT_MAX_TOKENS


class BERTContainer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
        self.bert_model = AutoModel.from_pretrained(BERT_MODEL)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device)
        self.bert_model.eval()


    def get_bert_embeddings(self, texts: List[str], pooling_strategy: str, batch_size=16) -> np.ndarray:
        print(f"Extracting embeddings using {pooling_strategy}")
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                # !IMPORTANT WHEN PROCESSING IN BATCHES DURING TRAIN/TEST AND PASSING SINGLE TEXTS DURING INFERENCE!
                padding='max_length',
                truncation=True,
                max_length=BERT_MAX_TOKENS,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                last_hidden = self.bert_model(**inputs).last_hidden_state
                if pooling_strategy == 'cls':
                    batch_embeddings = last_hidden[:, 0, :].cpu().numpy()  # [CLS] token
                elif pooling_strategy == 'mean':
                    batch_embeddings = last_hidden.mean(dim=1).cpu().numpy()  # Mean pooling
                elif pooling_strategy == 'max':
                    batch_embeddings = last_hidden.max(dim=1).values.cpu().numpy()  # Max pooling
                elif pooling_strategy == 'weighted_mean':
                    # Attention-weighted mean pooling
                    attention_mask = inputs['attention_mask']
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
                embeddings.append(batch_embeddings)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(texts)} texts")
        
        return np.vstack(embeddings)

    def enhance_embeddings(self, embeddings):
        """Add engineered features to BERT embeddings"""
        # Add magnitude of embeddings as a feature
        magnitudes = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add some statistical features
        mean_features = np.mean(embeddings, axis=1, keepdims=True)
        std_features = np.std(embeddings, axis=1, keepdims=True)
        
        return np.hstack([magnitudes, mean_features, std_features])
    
