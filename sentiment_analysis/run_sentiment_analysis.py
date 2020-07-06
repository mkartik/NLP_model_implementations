import argparse
import torch
import os
import pandas as pd
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from generate_dataset import preprocess_tweet

MAX_LEN = 60
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_CLASSES = 2


class TweetDataset(Dataset):
    def __init__(self, tweet, polarity, tokenizer, max_seq_length):
        self.tweets = tweet
        self.polarities = polarity
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        tweet = self.tweets[idx]
        polarity = self.polarities[idx]

        enc = self.tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=self.max_seq_length,
                                         return_token_type_ids=False, return_attention_mask=True,
                                         pad_to_max_length=True, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'tweet': tweet,
            'polarity': torch.tensor(polarity, dtype=torch.long)
        }


class TweetClassifier(nn.Module):
    def __init__(self, model):
        super(TweetClassifier, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(self.model.config.hidden_size, NUM_CLASSES)

    def forward(self, input_id, attention_mask):
        hidden_state, pool_output = self.model(input_ids=input_id, attention_mask=attention_mask)
        dropout_out = self.dropout(pool_output)
        out = self.linear(dropout_out)

        return out


def create_data_loader(df, tokenizer, max_length, batch_size):
    dataset = TweetDataset(df.tweet.to_numpy(), df.polarity.to_numpy(), tokenizer, MAX_LEN)
    return DataLoader(dataset, batch_size=batch_size, num_workers=4)


def train(model, dataloader, loss_fn, scheduler, optimizer, num_examples, device):
    model = model.train()
    total_loss = 0
    correct_predictions = 0
    for data in dataloader:
        input_ids, attention_mask, polarity = data['input_ids'].to(device), data['attention_mask'].to(device), data[
            'polarity'].to(device)
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)

        loss = loss_fn(outputs, polarity)
        correct_predictions += torch.sum(preds == polarity).double() / len(preds)

        total_loss += loss

        loss.backward()
        # print ("cur loss: ", loss)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / num_examples, total_loss.double() / num_examples


def eval(model, dataloader, loss_fn, num_examples, device):
    model = model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for data in dataloader:
            input_ids, attention_mask, polarity = data['input_ids'].to(device), data['attention_mask'].to(device), data[
                'polarity'].to(device)
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, polarity)
            correct_predictions += torch.sum(preds == polarity).double() / len(preds)
            total_loss += loss
    return correct_predictions.double() / num_examples, total_loss.double() / num_examples


def generate_predictions(model, dataloader, device):
    model = model.eval()
    tweets = []
    gold_value = []
    predicted_value = []
    with torch.no_grad():
        for data in dataloader:
            input_ids, attention_mask, polarity, tweet = data['input_ids'].to(device), data['attention_mask'].to(
                device), data['polarity'].to(device), data['tweet']
            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs, dim=1)

            tweets.extend(tweet)
            gold_value.extend(polarity)
            predicted_value.extend(preds)

    gold_value = torch.stack(gold_value).cpu()
    predicted_value = torch.stack(predicted_value).cpu()

    return tweets, gold_value, predicted_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir", type=str, help="directory containing the test/train/validation files",
    )
    parser.add_argument(
        "--no_cuda", default=False, type=bool, help="Whether to force execution on CPU.",
    )

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    df_train = pd.read_csv(os.path.join(args.data_dir, 'train.csv'))
    df_test = pd.read_csv(os.path.join(args.data_dir, 'test.csv'))
    df_val = pd.read_csv(os.path.join(args.data_dir, 'validation.csv'))

    train_dataloader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_dataloader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    val_dataloader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)

    bert_model = BertModel.from_pretrained('bert-base-cased')
    model = TweetClassifier(bert_model)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    loss_fn = nn.CrossEntropyLoss().to(args.device)

    # Training
    best_acc = 0
    for ep in range(NUM_EPOCHS):
        train_acc, train_loss = train(model, train_dataloader, loss_fn, scheduler, optimizer, len(train_dataloader),
                                      args.device)
        print("Epoch: ", ep)
        print("Training Accuracy: %s, Training Loss: %s" % (train_acc, train_loss))

        val_acc, val_loss = eval(model, val_dataloader, loss_fn, len(val_dataloader), args.device)
        print("Validation Accuracy: %s, Validation Loss: %s" % (val_acc, val_loss))

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.bin')

    # Test Set Evaluation
    model.load_state_dict(torch.load('best_model.bin'))
    model = model.to(args.device)

    test_acc, test_loss = eval(model, test_dataloader, loss_fn, len(test_dataloader), args.device)
    print("Test accuracy is: ", test_acc.item())

    #Generate Predictions
    tweets, y_gold, y_pred = generate_predictions(model, test_dataloader, args.device)

    # User Tweet
    test_tweet = "Great survey of the state-of-the-art language models https://medium.com/@phylypo/a-survey-of-the-state-of-the-art-language-models-up-to-early-2020-aba824302c6 #NLProc"
    tweet = preprocess_tweet(test_tweet)
    encoded_tweet = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=MAX_LEN,
                                          return_token_type_ids=False, return_attention_mask=True,
                                          pad_to_max_length=True, return_tensors='pt')

    input_ids, attention_mask = encoded_tweet['input_ids'].to(args.device), encoded_tweet['attention_mask'].to(args.device)
    outputs = model(input_ids, attention_mask)
    _, preds = torch.max(outputs, dim=1)

    polarities = ['negative', 'positive']
    print("User Tweet: ", test_tweet)
    print("Predicted polarity: ", polarities[preds])

