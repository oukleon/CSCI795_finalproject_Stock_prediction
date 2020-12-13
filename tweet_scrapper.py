import twint
import pandas as pd
import re


def mine_tweets(companies, tickers):
    # Configure
    cols = ['id', 'conversation_id', 'created_at', 'date', 'timezone', 'place',
            'tweet', 'language', 'hashtags', 'cashtags', 'user_id', 'user_id_str',
            'username', 'name', 'day', 'hour', 'link', 'urls', 'photos', 'video',
            'thumbnail', 'retweet', 'nlikes', 'nreplies', 'nretweets', 'quote_url',
            'search', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
            'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
            'trans_dest']
    tweeet_df = pd.DataFrame(columns=cols)
    for ticker in tickers:
        for company in companies:
            c = twint.Config()
            c.Username = company
            c.Search = ticker
            c.Pandas = True
            twint.run.Search(c)
            tweeet_df = tweeet_df.append(twint.storage.panda.Tweets_df, ignore_index=True)
    temp = tweeet_df["date"].str.split(" ", n=1, expand=True)
    tweeet_df.drop(columns=['date'], inplace=True)
    tweeet_df['date'] = temp[0]
    tweeet_df['time'] = temp[1]
    return tweeet_df


def df_to_csv(df):
    filename = "news_headlines.csv"
    df.to_csv(filename)


def run(companies, tickers):
    # df_obj = mine_tweets(["business", "reuters", "cnbc", "WSJmarkets","Benzinga"], ["AMD"])
    df_obj = mine_tweets(companies, tickers)
    new_df = df_obj.filter(['search', 'date', 'time', 'tweet'])
    new_df.rename(columns={'search': 'Company', 'tweet': 'headline'}, inplace=True)
    new_df['headline'] = new_df['headline'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])
    # new_df['date'] = pd.to_datetime(new_df['date'], format='%Y/%m/%d')
    # new_df.sort_values(by='date')
    df_to_csv(new_df)

