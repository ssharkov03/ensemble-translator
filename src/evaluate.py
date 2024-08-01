from collections import defaultdict

import numpy as np
from evaluate import load 


class SimilarityChecker:
    def __init__(
        self,
        score_names: list[str],
    ):  
        self.score_names = score_names
        assert all([score_name in ['bertscore'] for score_name in self.score_names])
        
        self.bertscorer = load("bertscore")

    def bertscore_ij(self, text_i, text_j):
        text_i = [text_i] if isinstance(text_i, str) else text_i
        text_j = [text_j] if isinstance(text_j, str) else text_j

        scores: dict = self.bertscorer.compute(predictions=text_i, references=text_j, lang="ru")
        response = {
            "precision": scores['precision'][0],
            "recall": scores['recall'][0],
            "f1": scores['f1'][0],
        }
        return response

    def group_bertscore(self, texts: list[str]):
        n_texts = len(texts)
        pairwise_scores = defaultdict(list)
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                bertscore_ij = self.bertscore_ij(texts[i], texts[j])
                for metric in bertscore_ij:
                    pairwise_scores[metric].append(bertscore_ij[metric])
        return {
            metric_name: np.mean(metric_pairwise_scores) for metric_name, metric_pairwise_scores in pairwise_scores.items()
        }

    def check_similarity(self, texts):
        similarity_scores = dict()
        for score_name in self.score_names:
            if score_name == "bertscore":
                similarity_scores[score_name] = self.group_bertscore(texts)
        return similarity_scores