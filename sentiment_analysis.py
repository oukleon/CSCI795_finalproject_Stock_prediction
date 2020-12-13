import pandas as pd
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import tweet_scrapper
from sklearn.utils import resample
from sklearn.naive_bayes import GaussianNB
import scrapping_finviz
import warnings


def read_create_train_test(filename):
    df = pd.read_csv(filename, encoding="cp1252")
    df_major = df[df['Sentiment'] == 1]
    df_minor = df[df['Sentiment'] == 0]
    df_upsized = resample(df_minor, replace=True, n_samples=551)
    df = pd.concat([df_major, df_upsized])
    temp = df.drop(['Company', 'time', 'date'], axis=1)
    newslines = temp['headline']
    labels = temp['Sentiment']
    newslines_train, newslines_test, y_train, y_test = train_test_split(newslines, labels, test_size=0.2)
    return newslines, labels, newslines_train, newslines_test, y_train, y_test, df


def vectorize_data(df, newslines, newslines_train, newslines_test):
    punctuations = string.punctuation
    parser = English()
    stopwords = list(STOP_WORDS)
    df.groupby('Sentiment').Company.count().plot.bar(ylim=0)
    plt.show()

    def spacy_tokenizer(utterance):
        tokens = parser(utterance)
        return [token.lemma_.lower().strip() for token in tokens if token.text.lower().strip()
                not in stopwords and token.text not in punctuations]

    # GNB (Works best with 1,1) broke with ngram while logistic regression worked better
    vectorized = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
    vectorized.fit(newslines)
    return vectorized.transform(newslines_train), vectorized.transform(newslines_test), vectorized


def create_sentiment(filename, vectorized, classifier):
    df = pd.read_csv(filename, encoding="utf8")
    X_new = vectorized.transform(df['headline'])
    sentiment = classifier.predict(X_new.A)
    df.insert(df.shape[1], "Sentiment", sentiment)
    df.to_csv(filename)


def run():
    warnings.filterwarnings('ignore')
    # tickers = ["AMD", "MSFT", "LMND"]
    news_headline_csvfile = "news_headlines.csv"
    # scrapping_finviz.run(tickers, news_headline_csvfile)
    tweet_scrapper.run(["business", "reuters", "cnbc", "WSJmarkets","Benzinga"], ["AMD"])
    training_filename = "training_data.csv"
    newslines, labels, newslines_train, newslines_test, y_train, y_test, df = \
        read_create_train_test(training_filename)
    X_train, X_test, vectorized = vectorize_data(df, newslines, newslines_train, newslines_test)
    logistic_reggression_clf = LogisticRegression()
    logistic_reggression_clf.fit(X_train.A, y_train.values)
    # gaussianNB_clf = GaussianNB()
    # gaussianNB_clf.fit(X_train.A, y_train.values)
    print("Accuracy for Logistic Regression:", logistic_reggression_clf.score(X_test.A, y_test.values))
    create_sentiment(news_headline_csvfile, vectorized, logistic_reggression_clf)


run()
"""
multinomial naive bayes
drop neutral sentiments
break multiple newslines
it's fine to reuse other people's code but learn from it
add heuristics. use patterns in the head news line to increase accuracy
test it statisically 
"""
