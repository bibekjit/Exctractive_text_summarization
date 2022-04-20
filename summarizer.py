import pandas as pd
from math import log
import os
import spacy
from numpy.linalg import svd
pd.options.mode.chained_assignment = None

MODEL_NAME = "en_core_web_md"
FILE_PATH = input('Enter path to text file : ')
SUMMARY_SIZE = input('Enter summary size : ')
docs = []


def text_to_tokens(text):
    global MODEL_NAME
    model = spacy.load(MODEL_NAME)
    tokens = model(text)
    tokens = [t for t in tokens if (not t.is_stop) and (not t.is_punct)]
    tokens = [str(t.lemma_).lower() for t in tokens if not t.is_space]
    tokens = [t for t in tokens if not t.isdigit()]
    return list(set(tokens))


def text_to_sentences(text):
    global MODEL_NAME
    global docs
    sentences = []

    model = spacy.load(MODEL_NAME)
    sent = ''

    for w in text.split():

        sent = sent + " " + w
        if (w[-1] == '.') and (not w[-2].isupper()):
            docs.append(sent.strip())
            sent_tokens = model(sent)
            sent_tokens = [str(t.lemma_).lower() for t in sent_tokens if not t.is_punct]
            sent = ' '.join(sent_tokens).strip()
            sentences.append(sent)
            sent = ''
    return sentences


def create_input_matrix(sentences, tokens):
    mat = pd.DataFrame(columns=[s for s in range(len(sentences))], index=tokens)
    mat.fillna(float(0), inplace=True)
    for t in mat.index:
        count = 0
        for i, sent in enumerate(sentences):
            if t in sent.split():
                count += 1

        idf = log(len(sentences) / (count + 1))

        for i, sent in enumerate(sentences):
            tf = sent.split().count(t) / len(sent.split())
            mat[i].loc[t] = tf * idf

    return mat


def summarize_text(input_matrix,summ_size):
    u, s, v = svd(input_matrix)

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
    sents = list(sent_scores)
    k = round(len(sents)*summ_size)
    summ_sents = sorted(sents[:k])

    summary = ""

    for i in summ_sents:
        summary = summary + " " + docs[i]

    return summary.strip()


def main(file_path, summary_size):

    with open(file_path,'r') as file:
        text = file.read().split('\n')
        text = ' '.join(text).strip()

    if summary_size == "":
        summary_size = 0.4
    else: summary_size = int(summary_size)/100
    tokens = text_to_tokens(text)
    sentences = text_to_sentences(text)
    input_matrix = create_input_matrix(sentences,tokens)
    summary = summarize_text(input_matrix,summary_size)

    filename = os.path.basename(file_path).split('.')[0]
    output_file_name = filename + '_summarized_' + str(int(summary_size * 100)) + ".txt"

    with open(output_file_name, "w") as f:
        f.write(summary)
        
    print('summary created !')


main(file_path=FILE_PATH,summary_size=SUMMARY_SIZE)

