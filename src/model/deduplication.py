import pandas as pd
import numpy as np
from typing import Tuple
from model.constants import BERT_DEDUPLICATOR_MODEL, CLUSTER1_PROPORTION_COL, CLUSTER_ID_COL, DATASET_SYNTHETIC_COLUMN, DATASET_TEXT_COLUMN, DATASET_CLASS_COLUMN, RANDOM_SEED
from sentence_transformers import SentenceTransformer, util



class SemanticDeduplicator:
    
    def __init__(self, augment_features = True):
        self.model = SentenceTransformer(BERT_DEDUPLICATOR_MODEL)
        self.augment_features = augment_features

    def clustering_deduplication(self, df: pd.DataFrame, threshold=0.8, pure_only = True, shuffle = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if threshold >= 1.0:
            raise ValueError("Threshold must be less than 1.0. For near-exact duplicates, use 0.999 or similar.")

        texts = df[DATASET_TEXT_COLUMN].tolist()
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        clusters = util.community_detection(embeddings, min_community_size=1, threshold=threshold)
        
        cluster_sizes = [len(cluster) for cluster in clusters]
        clustered_indices = set(idx for cluster in clusters for idx in cluster)
        print(f"\nUnclustered records: {len(df) - len(clustered_indices)}")
        print(f"Cluster sizes: {cluster_sizes}")
        print(f"Max cluster size: {max(cluster_sizes)}")
        print(f"Min cluster size: {min(cluster_sizes)}")

        unique_df, cluster_df = self.__deduplicate_clusters(clusters, df, pure_only)
        
        self.__print_cluster_stats(cluster_df)
        print(f"Deduplication reduced dataset from {len(df)} to {len(unique_df)}")
        
        if shuffle:
            unique_df = unique_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

        return unique_df, cluster_df
    
    def __deduplicate_clusters(self, clusters, df, pure_only):
        unique_data = []
        cluster_info = []

        def add_unique_record(class_indices, class_1_proportion, cluster_id):
            if DATASET_SYNTHETIC_COLUMN in df.columns:
                non_synthetic_indices = [idx for idx in class_indices if df.loc[idx, DATASET_SYNTHETIC_COLUMN] == 0]
                indices_to_consider = non_synthetic_indices if non_synthetic_indices else class_indices
            else:
                indices_to_consider = class_indices

            rep_idx = max(indices_to_consider, key=lambda idx: len(df.iloc[idx][DATASET_TEXT_COLUMN]))
            rep_row = df.iloc[rep_idx].copy()
            if (self.augment_features):
                rep_row[CLUSTER1_PROPORTION_COL] = class_1_proportion
                rep_row[CLUSTER_ID_COL] = cluster_id
            unique_data.append(rep_row.tolist())

        def add_all(class_indices, class_1_proportion, cluster_id):
            if DATASET_SYNTHETIC_COLUMN in df.columns:
                raise ValueError("Mixed clustering only supported for non-synthetic data.")
            reps = df.iloc[class_indices].copy()
            if (self.augment_features):
                reps[CLUSTER1_PROPORTION_COL] = class_1_proportion
                reps[CLUSTER_ID_COL] = cluster_id
            unique_data.extend(reps.values.tolist())

        
        for cluster_id, cluster in enumerate(clusters):
            class_0_indices = [idx for idx in cluster if df.iloc[idx][DATASET_CLASS_COLUMN] == 0]
            class_1_indices = [idx for idx in cluster if df.iloc[idx][DATASET_CLASS_COLUMN] == 1]
            class_1_proportion = len(class_1_indices) / len(cluster)
            
            if not class_0_indices:  # Only class 1
                add_unique_record(class_1_indices, class_1_proportion, cluster_id)
            elif not class_1_indices:  # Only class 0
                add_unique_record(class_0_indices, class_1_proportion, cluster_id)
            else:  # Mixed classes
                if pure_only:
                    # Add everithing
                    add_all(class_0_indices, class_1_proportion, cluster_id)
                    add_all(class_1_indices, class_1_proportion, cluster_id)
                else:
                    # Add a representative of each class
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

        cols = df.columns.tolist()
        if self.augment_features:
            cols.extend([CLUSTER1_PROPORTION_COL, CLUSTER_ID_COL])
        return pd.DataFrame(unique_data, columns=cols), pd.DataFrame(cluster_info)

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