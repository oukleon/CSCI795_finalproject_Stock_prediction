import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import scrapping_finviz
import warnings


def read_create_train_test(filename):
    df = pd.read_csv(filename, encoding="cp1252")
    temp = df.drop(['Company', 'time', 'date'], axis=1)
    newslines = temp['headline']
    labels = temp['Sentiment']
    newslines_train, newslines_test, y_train, y_test = train_test_split(newslines, labels, test_size=0.2)
    return newslines, labels, newslines_train, newslines_test, y_train, y_test, df


def vectorize_training_data(df, newslines, newslines_train, newslines_test):
    punctuations = string.punctuation
    parser = English()
    stopwords = list(STOP_WORDS)
    df.groupby('Sentiment').Company.count().plot.bar(ylim=0)
    plt.show()

    def spacy_tokenizer(utterance):
        tokens = parser(utterance)
        return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip()
                not in stopwords and token.text not in punctuations]

    vectorized = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
    vectorized.fit(newslines)
    return vectorized.transform(newslines_train), vectorized.transform(newslines_test), vectorized


def real_data(filename):
    df = pd.read_csv(filename, encoding="cp1252")


def run():
    warnings.filterwarnings('ignore')
    tickers = ["AMD", "MSFT", "LMND"]
    create_csv_name = "news_headlines.csv"
    scrapping_finviz.run(tickers, create_csv_name)
    training_filename = "training_data.csv"
    newslines, labels, newslines_train, newslines_test, y_train, y_test, df = \
        read_create_train_test(training_filename)
    X_train, X_test, vectorized = vectorize_training_data(df, newslines, newslines_train, newslines_test)
    news_headlines = ['Apple releases new Macbook aimed towards older generations',
                      'Apple Everything Store Is Now Everyones Antitrust Target']
    X_new = vectorized.transform(news_headlines)
    gaussianNB_clf = GaussianNB()
    gaussianNB_clf.fit(X_train.A, y_train.values)
    print(gaussianNB_clf.predict(X_new.A))


run()
# # encoding is important since it seems like csv contains some hectic encoding
# df = pd.read_csv('news_headlines.csv', encoding="cp1252")
# temp = df.drop(['Company', 'time', 'date'], axis=1)
#
# newslines = temp['headline']
# labels = temp['Sentiment']
#
# newslines_train, newslines_test, y_train, y_test = train_test_split(newslines, labels,
#                                                                     test_size=0.2)
#
# punctuations = string.punctuation
# parser = English()
# stopwords = list(STOP_WORDS)
# df.groupby('Sentiment').Company.count().plot.bar(ylim=0)
# plt.show()
#
#
# def spacy_tokenizer(utterance):
#     tokens = parser(utterance)
#     return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip()
#             not in stopwords and token.text not in punctuations]
#
#
# vectorized = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
# vectorized.fit(newslines)

# X_train = vectorized.transform(newslines_train)
# X_test = vectorized.transform(newslines_test)
# logistic_reggression_clf = LogisticRegression()
# logistic_reggression_clf.fit(X_train, y_train)
#
# print(logistic_reggression_clf.score(X_test, y_test))
#
# news_headlines = ['Apple releases new Macbook aimed towards older generations',
#                   'Apple Everything Store Is Now Everyones Antitrust Target']
# X_new = vectorized.transform(news_headlines)
# print(logistic_reggression_clf.predict(X_new))
# gaussianNB_clf = GaussianNB()
# gaussianNB_clf.fit(X_train.A, y_train.values)
# print(gaussianNB_clf.score(X_test.A, y_test.values))
# news_headlines = ['Apple releases new Macbook aimed towards older generationse',
#                   'Apple Everything Store Is Now Everyones Antitrust Target']
# X_new = vectorized.transform(news_headlines)
# print(gaussianNB_clf.predict(X_new.A))
"""
multinomial naive bayes
drop neutral sentiments
break multiple newslines
it's fine to reuse other people's code but learn from it
add heuristics. use patterns in the head news line to increase accuracy
test it statisically 
"""
