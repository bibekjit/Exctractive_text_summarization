import pandas as pd
from math import log
import spacy
from numpy.linalg import svd
pd.options.mode.chained_assignment = None


def preprocessing(text, model='en_core_web_md'):
    model = spacy.load(model)
    data = model(text)
    tokens = [t for t in data if (not t.is_stop) and (not t.is_punct)]
    tokens = [str(t.lemma_).lower() for t in tokens if not t.is_space]
    tokens = [t for t in tokens if not t.isdigit()]
    tokens = list(set(tokens))
    docs = [str(sent.text) for sent in data.sents]
    sentences = [str(sent.text).lower() for sent in data.sents]
    return tokens,docs,sentences


class SVDSummarizer:
    def __init__(self, summary_size=None):
        self.summary_size = summary_size
        self.input_matrix = None

    def fit(self,sentences,tokens):
        self.input_matrix = pd.DataFrame(columns=[s for s in range(len(sentences))], index=tokens)
        self.input_matrix.fillna(float(0), inplace=True)
        for t in self.input_matrix.index:
            count = 0
            for i, sent in enumerate(sentences):
                if t in sent.split():
                    count += 1
            idf = log(len(sentences) / (count + 1))
            for i, sent in enumerate(sentences):
                tf = sent.split().count(t) / len(sent.split())
                self.input_matrix[i].loc[t] = tf * idf

    def text_summary(self,docs):
        u, s, v = svd(self.input_matrix)
        print(v.shape)
        if self.summary_size is None:
            self.summary_size = int(len(v)*0.6)

        v = v[:, self.summary_size+1]
        print(v.shape)
        # for i in range(len(v.T)):
        #     row = v.T[i]
        #     mean = sum(row) / len(row)
        #     for j, val in enumerate(row):
        #         if val <= mean:
        #             row[j] = 0
        #
        # sent_scores = {}
        # for i in range(len(v)):
        #     if sum(v[i]) > 0:
        #         sent_scores[i] = sum(v[i])
        # sent_scores = {k: v for k, v in sorted(sent_scores.items(), key=lambda item: item[1], reverse=True)}
        # summ_sents = list(sent_scores)
        # summary = ""
        #
        # for i in summ_sents:
        #     summary = summary + " " + docs[i]
        # return summary.strip()









