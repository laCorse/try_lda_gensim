import nltk
import nltk.data
import numpy as np
import pandas as pd
import re
from gensim.models.ldamodel import LdaModel
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary


def clean_text(text):
    # 数据清洗
    text = text.replace('\n', " ")
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\d+/\d+/\d+", "", text)
    text = re.sub(r"[0-2]?[0-9]:[0-6][0-9]", "", text)
    text = re.sub(r"[\w]+@[\.\w]+", "", text)
    text = re.sub(r"/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)  # 网址，没意义
    pure_text = ''
    for letter in text:
        if letter.isalpha() or letter == ' ':
            pure_text += letter
    text = ' '.join(word for word in pure_text.split() if len(word) > 1)
    return text

def splitSentence(paragraph):
    tokenizer = nltk.data.load('punkt/english.pickle')
    sentences = tokenizer.tokenize(paragraph)
    return sentences


if __name__ == '__main__':

    df = pd.read_csv("./total_info.txt",sep='，',header=0,names=['A','B','C','D','E','F'])
    data = df['B']
    data = data.apply(lambda s:clean_text(s))
    datalist = data.values
    print(datalist)
    # 分词
    texts = [[word for word in doc.lower().split() ] for doc in datalist]
    print(texts[0])

    common_dictionary = Dictionary(texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in texts]
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=20)
    print(lda.print_topic(10, topn=5))

    lda.save('lda.model')
    lda = LdaModel.load('lda.model')


    tryTxt = "while i be suffer i be able to press and go in subscribe but when i press the video it keep show no connection ."
    trylist = [word for word in tryTxt.lower().split()]
    bow = common_dictionary.doc2bow(trylist)
    print(lda.get_document_topics(bow))

    import pyLDAvis.gensim
    # 浏览器打开http://127.0.0.1:8888/
    vis = pyLDAvis.gensim.prepare(lda, common_corpus, common_dictionary)
    pyLDAvis.show(vis)

