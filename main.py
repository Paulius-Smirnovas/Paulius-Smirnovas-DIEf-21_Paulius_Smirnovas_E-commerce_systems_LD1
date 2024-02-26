import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import os
import re
from transformers import Trainer, TrainingArguments
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from sklearn.metrics import classification_report
import numpy as np
from transformers import DistilBertTokenizerFast

def prepare_data(data_path):
    """Prepare the data for analysis."""
    data = pd.read_csv(data_path)
    data['cleaned_text'] = data['review'].apply(clean_text)
    return data


def clean_text(text):
    """Clean the text."""
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # numbers are not removed
    return text.lower()


def encode_labels(labels):
    return [1 if sentiment == 'positive' else 0 for sentiment in labels]


def train_and_evaluate_lr(data):
    """Train and evaluate the Logistic Regression model."""
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['cleaned_text'])
    y = encode_labels(data['sentiment'].values)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Save the TF-IDF Vectorizer
    dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

    # Check if the model already exists
    if os.path.exists('best_model.pkl'):
        print("The model 'best_model.pkl' already exists. Loading the model and evaluating.")
        grid = load('best_model.pkl')
    else:
        # Hyperparameter tuning
        param_grid = {'C': [0.1, 1, 10], 'max_iter': [500, 1000, 2000]}
        grid = GridSearchCV(LogisticRegression(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)

        # Save the best model if cross-validation score is above 0.8
        if grid.best_score_ > 0.8:
            dump(grid, 'best_model.pkl')

    y_pred = grid.predict(X_test)
    # Model evaluation
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


def train_and_evaluate_rf(data):
    """Train and evaluate the Random Forest model."""
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X = tfidf_vectorizer.fit_transform(data['cleaned_text'])
    y = encode_labels(data['sentiment'].values)

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Check if the model already exists
    if os.path.exists('best_rf_model.pkl'):
        print("The model 'best_rf_model.pkl' already exists. Loading the model and evaluating.")
        grid = load('best_rf_model.pkl')
    else:
        # Hyperparameter tuning
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
        grid = GridSearchCV(RandomForestClassifier(), param_grid, refit=True, verbose=3)
        grid.fit(X_train, y_train)

        # Save the best model if cross-validation score is above 0.8
        if grid.best_score_ > 0.8:
            dump(grid, 'best_rf_model.pkl')

    y_pred = grid.predict(X_test)
    # Model evaluation
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))


def evaluate_bert(trainer, tokenizer, val_texts, val_labels):
    """Evaluate the BERT model."""
    # Convert validation data into torch tensors
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    val_dataset = IMDbDataset(val_encodings, val_labels)

    # Predict on validation data
    predictions, _, _ = trainer.predict(val_dataset)
    predictions = np.argmax(predictions, axis=1)

    # Print classification report
    print(classification_report(val_labels, predictions))


def train_and_evaluate_bert(data):
    """Train and evaluate the BERT model."""
    # Prepare data for BERT
    sentences = data['cleaned_text'].tolist()
    labels = encode_labels(data['sentiment'].tolist())

    # Load the DistilBERT tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Split data into training and validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(sentences, labels, test_size=0.2,
                                                                        random_state=42)

    # Limit the amount of data to speed up training
    train_texts = train_texts[:5000]
    train_labels = train_labels[:5000]

    # Convert data into torch tensors
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    class IMDbDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    train_dataset = IMDbDataset(train_encodings, train_labels)
    val_dataset = IMDbDataset(val_encodings, val_labels)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
    )

    if os.path.exists('./distilbert_model'):
        print("The model 'distilbert_model' already exists. Loading the model.")
        model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
    else:
        # Load DistilBERT model
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        # Train model
        trainer.train()

        # Save model
        model.save_pretrained('./distilbert_model')

    # Create trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    # Evaluate on validation data
    evaluate_bert(trainer, tokenizer, val_texts, val_labels)


def flag_reviews_for_response(data):
    """Flag reviews that require a response based on certain criteria."""

    # Load the trained Logistic Regression model and TF-IDF Vectorizer
    model = load('best_model.pkl')
    tfidf_vectorizer = load('tfidf_vectorizer.pkl')

    # Define the keywords that indicate a response is necessary
    keywords = ['refund', 'return', 'broken', 'not working', 'disappointed', 'complaint',
                'bad acting', 'poor quality', 'waste of time', 'misleading', 'false advertising',
                'copyright', 'infringement', 'plagiarism', 'illegal', 'lawsuit', 'sue', 'rights',
                'permission', 'unauthorized', 'defamation']

    # Set the sentiment score threshold for negative sentiment
    negative_sentiment_threshold = 0.5  # assuming the sentiment scores range from 0 to 1

    # Set the review length threshold
    length_threshold = 200

    X = tfidf_vectorizer.transform(data['cleaned_text'])
    sentiment_scores = model.predict_proba(X)[:, 1]

    # Create a new column 'response_flag' that indicates whether a response is necessary
    data['response_flag'] = np.where(
        (sentiment_scores < negative_sentiment_threshold) |
        (data['review'].str.len() > length_threshold) |
        (data['review'].str.contains('|'.join(keywords), case=False, na=False)),
        'Response Needed',
        'No Response Needed'
    )

    # Save to separate CSV files
    data[data['response_flag'] == 'Response Needed'].to_csv('reviews_needing_response.csv', index=False)
    data[data['response_flag'] == 'No Response Needed'].to_csv('reviews_not_needing_response.csv', index=False)

    return data

def is_response_needed(review):
    """Determine whether a response is needed for a given review."""
    # Load the trained Logistic Regression model and TF-IDF Vectorizer
    model = load('best_model.pkl')
    tfidf_vectorizer = load('tfidf_vectorizer.pkl')

    # Define the keywords that indicate a response is necessary
    keywords = ['refund', 'return', 'broken', 'not working', 'disappointed', 'complaint',
                'bad acting', 'poor quality', 'waste of time', 'misleading', 'false advertising',
                'copyright', 'infringement', 'plagiarism', 'illegal', 'lawsuit', 'sue', 'rights',
                'permission', 'unauthorized', 'defamation']

    # Set the sentiment score threshold for negative sentiment
    negative_sentiment_threshold = 0.5  # assuming the sentiment scores range from 0 to 1

    # Set the review length threshold
    length_threshold = 200

    # Clean the text
    review = clean_text(review)
    X = tfidf_vectorizer.transform([review])
    sentiment_score = model.predict_proba(X)[:, 1][0]

    # Check whether the review meets any of the criteria
    if (sentiment_score < negative_sentiment_threshold or
        len(review) > length_threshold or
        any(keyword in review for keyword in keywords)):
        print("The review is worthy of a response.")
    else:
        print("The review is not worthy of a response.")



def predict_sentiment(review):
    """Predict the sentiment of a review using all three models."""
    # Load the trained Logistic Regression, Random Forest, and BERT models
    lr_model = load('best_model.pkl')
    rf_model = load('best_rf_model.pkl')
    bert_model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Load the TF-IDF Vectorizer
    tfidf_vectorizer = load('tfidf_vectorizer.pkl')

    # Clean the text and convert it into a format that can be used by the models
    review = clean_text(review)
    X = tfidf_vectorizer.transform([review])

    # Predict with the Logistic Regression model
    lr_prob = lr_model.predict_proba(X)[:, 1][0]
    lr_pred = 'positive' if lr_prob > 0.5 else 'negative'
    print(f"Logistic Regression: Probability={lr_prob:.2f}, Prediction={lr_pred}")

    # Predict with the Random Forest model
    rf_prob = rf_model.predict_proba(X)[:, 1][0]
    rf_pred = 'positive' if rf_prob > 0.5 else 'negative'
    print(f"Random Forest: Probability={rf_prob:.2f}, Prediction={rf_pred}")

    # Predict with the BERT model
    # Note: For the BERT model, we need to convert the text into a format that the model can understand
    inputs = tokenizer(review, return_tensors='pt')
    outputs = bert_model(**inputs)
    bert_prob = torch.nn.functional.softmax(outputs.logits, dim=-1).tolist()[0][1]
    bert_pred = 'positive' if bert_prob > 0.5 else 'negative'
    print(f"BERT: Probability={bert_prob:.2f}, Prediction={bert_pred}")
def main():
    """Main function."""
    path = r"C:\Users\pauli\Desktop\Elektroninio verslo sistemu laborai\IMDB Dataset.csv"
    data = prepare_data(path)
    train_and_evaluate_lr(data)
    train_and_evaluate_rf(data)
    #uncomment this if need to retrain model
    #train_and_evaluate_bert(data)
    data = flag_reviews_for_response(data)

    # Test with your own reviews
    while True:
        review = input("Enter a review (or '0' to stop): ")
        if review == '0':
            break
        is_response_needed(review)


    # Test with your own reviews
    while True:
        review = input("Enter a review (or '0' to stop): ")
        if review == '0':
            break
        predict_sentiment(review)



if __name__ == "__main__":
    main()

