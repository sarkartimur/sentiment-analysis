import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util


CLUSTER_ID_COL = 'cluster_id'
CLUSTER1_PROPORTION_COL = 'cluster_class_1_proportion'


class SemanticDeduplicator:
    
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', text_column='text', class_column='label'):
        self.model = SentenceTransformer(model_name)
        self.text_column = text_column
        self.class_column = class_column

    def clustering_deduplication(self, df, threshold=0.8):
        if threshold >= 1.0:
            raise ValueError("Threshold must be less than 1.0. For near-exact duplicates, use 0.999 or similar.")

        texts = df[self.text_column].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        clusters = util.community_detection(embeddings, min_community_size=1, threshold=threshold)
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        clustered_indices = set(idx for cluster in clusters for idx in cluster)
        print(f"\nUnclustered records: {len(df) - len(clustered_indices)}")
        print(f"Cluster sizes: {cluster_sizes}")
        print(f"Max cluster size: {max(cluster_sizes)}")
        print(f"Min cluster size: {min(cluster_sizes)}")

        unique_df, cluster_df = self.__deduplicate_clusters(clusters, df)
        
        self.__print_cluster_stats(cluster_df)
        print(f"Deduplication reduced dataset from {len(df)} to {len(unique_df)}")
        
        return unique_df, cluster_df
    
    def __deduplicate_clusters(self, clusters, df):
        unique_data = []
        cluster_info = []

        def add_unique_record(class_indices, class_1_proportion, cluster_id):
            rep_idx = max(class_indices, key=lambda idx: len(df.iloc[idx][self.text_column]))
            rep_row = df.iloc[rep_idx].copy()
            rep_row[CLUSTER1_PROPORTION_COL] = class_1_proportion
            rep_row[CLUSTER_ID_COL] = cluster_id
            unique_data.append(rep_row)
        
        for cluster_id, cluster in enumerate(clusters):
            class_0_indices = [idx for idx in cluster if df.iloc[idx][self.class_column] == 0]
            class_1_indices = [idx for idx in cluster if df.iloc[idx][self.class_column] == 1]
            class_1_proportion = len(class_1_indices) / len(cluster)
            
            if not class_0_indices:  # Only class 1
                add_unique_record(class_1_indices, class_1_proportion, cluster_id)
            elif not class_1_indices:  # Only class 0
                add_unique_record(class_0_indices, class_1_proportion, cluster_id)
            else:  # Mixed classes
                # Take both representatives
                add_unique_record(class_0_indices, class_1_proportion, cluster_id)
                add_unique_record(class_1_indices, class_1_proportion, cluster_id)
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'size': len(cluster),
                'class_1_proportion': class_1_proportion,
                'class_0_count': len(class_0_indices),
                'class_1_count': len(class_1_indices),
                'original_cluster': cluster,
                'class_0_indices': class_0_indices,
                'class_1_indices': class_1_indices
            })

        return pd.DataFrame(unique_data), pd.DataFrame(cluster_info)

    def __print_cluster_stats(self, cluster_df):
        mixed_clusters = len(cluster_df[(cluster_df['class_0_count'] > 0) & (cluster_df['class_1_count'] > 0)])
        class_0_only = len(cluster_df[(cluster_df['class_0_count'] > 0) & (cluster_df['class_1_count'] == 0)])
        class_1_only = len(cluster_df[(cluster_df['class_1_count'] > 0) & (cluster_df['class_0_count'] == 0)])

        print(f"\nCluster statistics:")
        print(f"Total clusters: {len(cluster_df)}")
        print(f"Mixed-class clusters: {mixed_clusters}")
        print(f"Class 0-only clusters: {class_0_only}")
        print(f"Class 1-only clusters: {class_1_only}")
        print(f"Average class 1 proportion: {cluster_df['class_1_proportion'].mean():.3f}")