from sklearn.cluster import AgglomerativeClustering
import numpy as np
import shap
from model.protocols import BERTWrapperMixin


class BERTExplainer:
    def __init__(self, bert_wrapper: BERTWrapperMixin, predict_method, cluster_size=10):
        self.__predict_method = predict_method
        self.__bert_wrapper = bert_wrapper
        self.__cluster_size = cluster_size


    def explain_prediction(self, text: str, max_evals = 1000):
        masker = shap.maskers.Text(
            tokenizer=self._semantic_tokenizer,
            mask_token='[MASK]',
            collapse_mask_token=True
        )
        
        explainer = shap.Explainer(
            model=self.__predict_method,
            masker=masker
        )
        
        shap_values = explainer([text], max_evals=max_evals)
        return shap_values

    def _semantic_tokenizer(self, text, return_offsets_mapping=True):
        inputs, tokens = self.__bert_wrapper.get_tokens_with_offsets(text, return_offsets_mapping=return_offsets_mapping)
        
        offsets_mapping = inputs['offset_mapping'][0].tolist()
        clusters = self._create_attention_clustering(inputs, tokens)
        input_ids = []
        offset_mapping = []
        for cluster in clusters:
            cluster_offsets = [offsets_mapping[idx] for idx in cluster]
            start_char = cluster_offsets[0][0]
            end_char = cluster_offsets[-1][1]
            text_chunk = text[start_char:end_char].strip()
            if text_chunk:
                input_ids.append(text_chunk)
                offset_mapping.append((start_char, end_char))
        
        return {
            "input_ids": input_ids,
            "offset_mapping": offset_mapping
        }

    def _create_attention_clustering(self, inputs, tokens):
        # Filter out special tokens
        valid_indices = []
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                valid_indices.append(i)
        
        if len(valid_indices) < 2:
            return [[i] for i in range(len(tokens))]
        
        # Build similarity matrix
        attention_matrix = self.__bert_wrapper.get_attention_matrix(inputs)
        n_valid = len(valid_indices)
        similarity_matrix = np.zeros((n_valid, n_valid))
        for i, idx_i in enumerate(valid_indices):
            for j, idx_j in enumerate(valid_indices):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = (attention_matrix[idx_i, idx_j] + attention_matrix[idx_j, idx_i]) / 2
                    similarity_matrix[i, j] = sim
        
        # Semantic clustering by distance
        clustering = AgglomerativeClustering(
            n_clusters=int(len(valid_indices)/self.__cluster_size),
            metric='precomputed',
            linkage='average'
        )
        distance_matrix = 1 - similarity_matrix
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Create cluster groups
        clusters = {}
        for label, token_idx in zip(cluster_labels, valid_indices):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(token_idx)
        
        # Singleton clusters for special tokens
        clustering_result = list(clusters.values())
        for i, token in enumerate(tokens):
            if i not in valid_indices:
                clustering_result.append([i])
        
        return clustering_result