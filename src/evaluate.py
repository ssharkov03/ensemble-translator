from collections import defaultdict

import spacy 
import numpy as np
from evaluate import load 
from nltk.translate.chrf_score import sentence_chrf


class SimilarityChecker:
    def __init__(
        self,
        score_names: list[str],
    ):  
        self.score_names = score_names
        assert all([score_name in ['bertscore', 'sentence_chrf'] for score_name in self.score_names])
        
        self.bertscorer = load("bertscore")
        self.nlp = spacy.load('ru_core_news_sm')


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

    def chrf_ij(self, text_i, text_j):
        text_i = text_i[0] if isinstance(text_i, list) else text_i
        text_j = text_j[0] if isinstance(text_j, list) else text_j
        score: float = sentence_chrf(
            hypothesis=self.normalize_text(text_i, self.nlp),
            reference=self.normalize_text(text_j, self.nlp),
        )
        response = {
            "value": score,
        }
        return response

    @staticmethod
    def is_punctuation(s: str) -> bool:
        characters = list(set(s))
        if len(characters) == 0:
            return False
        ok = True
        for c in characters:
            if c.isalnum():
                ok = False
                break
        return ok
    
    def normalize_text(self, s: str, spacy_nlp) -> str:
        doc = spacy_nlp(s)
        normalized = ' '.join(filter(
            lambda it2: (not self.is_punctuation(it2)) and (len(it2) > 0),
            map(lambda it1: it1.lemma_.lower(), doc)
        )).strip()
        del doc
        return normalized


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

    def group_chrf(self, texts: list[str]):
        n_texts = len(texts)
        pairwise_scores = defaultdict(list)
        for i in range(n_texts):
            for j in range(i + 1, n_texts):
                chrf_ij = self.chrf_ij(texts[i], texts[j])
                for metric in chrf_ij:
                    pairwise_scores[metric].append(chrf_ij[metric])
        return {
            metric_name: np.mean(metric_pairwise_scores) for metric_name, metric_pairwise_scores in pairwise_scores.items()
        }

    def check_similarity(self, texts):
        similarity_scores = dict()
        for score_name in self.score_names:
            if score_name == "bertscore":
                similarity_scores[score_name] = self.group_bertscore(texts)
            elif score_name == "sentence_chrf":
                similarity_scores[score_name] = self.group_chrf(texts)
        return similarity_scores

   