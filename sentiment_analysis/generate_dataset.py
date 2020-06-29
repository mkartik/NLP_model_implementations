import argparse
import pandas as pd
import re
import os
import html
from sklearn.model_selection import train_test_split

def update_polarity(polarity):
  if polarity==0:
    return 0
  else:
    return 1

def preprocess_tweet(text):
    # remove HTML tags
    def remove_html_tags(sentence):
        sentence = html.unescape(sentence)
        sentence = sentence.replace("\\", "")
        sentence = sentence.replace("\r", " ")
        sentence = sentence.replace("\n", " ")

        p = re.compile(r'\/\w+\/')
        sentence = re.sub(p, ' ', sentence)

        p = re.compile('<.*?>')
        sentence = re.sub(p, ' ', str(sentence))

        sentence = re.sub(r'http\S+', '', sentence)
        return sentence

    text = remove_html_tags(text)

    # remove @ Instances
    p = re.compile('@.*?\s+')
    text = re.sub(p, '', text)

    return text


if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dataset", type=str, help="input dataset file (.csv)",
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_dataset, header=None,
                     names=['polarity', 'id', 'date', 'query', 'user', 'tweet'], encoding='latin-1')

    df['polarity'] = df.polarity.apply(update_polarity)
    df['tweet'] = df.tweet.apply(preprocess_tweet)

    # Sampling 100000 entries (resource constraints)
    # df = df.sample(100000)

    df_train, df_val = train_test_split(df, test_size=0.2)
    df_val, df_test = train_test_split(df_val, test_size=0.5)

    data_dir = 'data'

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df_train.to_csv(os.path.join(data_dir, "train.csv"), sep=',', columns=['polarity', 'tweet'], index=False)
    df_val.to_csv(os.path.join(data_dir, "validation.csv"), sep=',', columns=['polarity', 'tweet'], index=False)
    df_test.to_csv(os.path.join(data_dir, "test.csv"), sep=',', columns=['polarity', 'tweet'], index=False)