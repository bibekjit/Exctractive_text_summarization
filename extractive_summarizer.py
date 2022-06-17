import pandas as pd
from math import log
import spacy
from numpy.linalg import svd
pd.options.mode.chained_assignment = None
import numpy as np


def preprocessing(text, model='en_core_web_md'):
    model = spacy.load(model)
    data = model(text)
    tokens = [t for t in data if (not t.is_stop) and (not t.is_punct)]
    tokens = [str(t.lemma_).lower() for t in tokens if not t.is_space]
    tokens = [t for t in tokens if not t.isdigit()]
    tokens = list(set(tokens))
    docs = [str(sent.text) for sent in data.sents]
    sentences = [str(sent.text).lower().strip() for sent in data.sents]
    sentences = [sent for sent in sentences if sent != '']
    return tokens,docs,sentences


class SVDSummarizer:
    def __init__(self, summary_size=2):
        self.summary_size = summary_size
        self.summary = None
        self.input_matrix = None

    def fit(self, tokens, sentences):
        self.input_matrix = np.zeros((len(tokens), len(sentences)), dtype=np.float64)
        for t in range(len(self.input_matrix)):
            count = 0
            for sent in sentences:
                if tokens[t] in sent.split():
                    count += 1
            idf = log(len(sentences) / (count + 1))
            for i, sent in enumerate(sentences):
                tf = sent.split().count(tokens[t]) / len(sent.split())
                self.input_matrix[t][i] = tf * idf

    def summarize(self,docs):
        u, s, v = svd(self.input_matrix)
        v = v[:, :self.summary_size]

        for i in range(len(v.T)):
            row = v.T[i]
            mean = sum(row) / len(row)
            for j, val in enumerate(row):
                if val <= mean:
                    row[j] = 0

        sent_scores = {}
        for i in range(len(v)):
            if sum(v[i]) > 0:
                sent_scores[i] = sum(v[i])
        sent_scores = {k: v for k, v in sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)}
        summ_sents = list(sent_scores)

        self.summary = [docs[i] for i in summ_sents]












