# Sentiment Analysis
A Tweet sentiment classifier, that predicts whether a Tweet has positive/negative sentiment, using HuggingFace Transformers.

## Getting Started

### Download the dataset (Sentiment140)
```
wget -nc http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

unzip trainingandtestdata.zip
```
We would we using the file 'training.1600000.processed.noemoticon.csv' containing 1600000 tweets as our dataset.

More details about the dataset can be obtained at http://help.sentiment140.com/for-students

### Setting up the enviroment
```
pip install -r requirements.txt
```

## Pre-process the dataset
```
python generate_dataset.py training.1600000.processed.noemoticon.csv
```

## Train the model and evaluate
```
python run_sentiment_analysis data/
```

IPython Notebook, sentiment_analysis.ipynb can also be used for the above steps.

