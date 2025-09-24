import pandas as pd
import numpy as np
import data_loader as dl
from sentence_transformers import SentenceTransformer, util


class SemanticDeduplicator:
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def clustering_deduplication(self, df, threshold=0.8):
        texts = df.tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        clusters = util.community_detection(embeddings, min_community_size=1, threshold=threshold)
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(f"\nCluster sizes: {cluster_sizes}")
        print(f"Max cluster size: {max(cluster_sizes)}")
        print(f"Min cluster size: {min(cluster_sizes)}")

        unique_data = []
        for _, cluster in enumerate(clusters):
            # Use longest text as representative
            rep_idx = max(cluster, key=lambda idx: len(texts[idx]))
            
            rep_row = df.iloc[rep_idx]
            unique_data.append(rep_row)
        
        unique_df = pd.DataFrame(unique_data)

        print(f"Deduplication reduced dataset from {len(df)} to {len(unique_df)}")
        
        return clusters, unique_df


dl = dl.DataLoader()
X_train, y_train, X_test, y_test = dl.load_data(sample_size=2000)
sd = SemanticDeduplicator()
clusters, unique_df = sd.clustering_deduplication(X_train)
