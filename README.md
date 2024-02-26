# Paulius-Smirnovas-DIEf-21_Paulius_Smirnovas_E-commerce_systems_LD1
This repository contains lab work from for E-commerce systems by Paulius Smirnovas from group DIEf-21. 


# Sentiment Analysis on User Reviews

## Project Overview

This project is aimed at building a sentiment analysis tool that can classify user reviews as positive or negative. It uses three different machine learning models for sentiment classification: Logistic Regression, Random Forest, and BERT. The tool also flags certain reviews that require a response, based on various criteria like sentiment score, presence of specific keywords, and the length of the review.

## Code Structure

The code is structured into several functions, each performing a specific task:

1. `prepare_data(data_path)`: This function loads and cleans the data.
2. `clean_text(text)`: This function removes unnecessary characters from the review text and converts it to lowercase.
3. `encode_labels(labels)`: This function encodes the sentiment labels as 0 (negative) and 1 (positive).
4. `train_and_evaluate_lr(data)`: This function trains and evaluates a Logistic Regression model.
5. `train_and_evaluate_rf(data)`: This function trains and evaluates a Random Forest model.


## Code explanation

1. prepare_data(data_path) Function:

This function is responsible for loading and preparing your data for the analysis. It uses the pandas read_csv function to load the data from a CSV file. Then, it applies the clean_text function to the 'review' column of the data and stores the results in a new column, 'cleaned_text'.

2. clean_text(text) Function:

This function cleans the input text by removing HTML tags and non-alphanumeric characters (except spaces), and converting the text to lowercase. It uses the re.sub function from the re module, which replaces substrings that match a regular expression.

3. encode_labels(labels) Function:

This function converts the sentiment labels from 'positive' and 'negative' to 1 and 0, respectively. This is a necessary preprocessing step because machine learning models require numeric inputs.

4. train_and_evaluate_lr(data) Function:

This function trains and evaluates a Logistic Regression model.

First, it transforms the 'cleaned_text' column into TF-IDF vectors using the TfidfVectorizer class from scikit-learn. It limits the number of features to 5000 to prevent the feature space from becoming too large.
The sentiment labels are then encoded using the encode_labels function.
The data is split into a training set, a validation set, and a test set using the train_test_split function from scikit-learn.
The trained TF-IDF vectorizer is saved for future use using the dump function from the joblib module.
Then, it checks if a model named 'best_model.pkl' already exists. If it does, it loads the model using the load function from the joblib module and uses it to predict the labels for the test set. If the model doesn't exist, it performs hyperparameter tuning using grid search, trains the best model on the training data, and saves it if its cross-validation score is above 0.8.
Finally, it prints a classification report for the test data, which includes metrics such as precision, recall, and F1-score.
5. train_and_evaluate_rf(data) Function:

This function is similar to the train_and_evaluate_lr function but trains and evaluates a Random Forest model instead of a Logistic Regression model. The steps are mostly the same, except for the hyperparameters used for grid search and the model used for training and prediction.

The results of the model evaluations are printed out in the form of classification reports. These reports provide detailed information about the performance of the models, including precision, recall, and F1-score for both the 'positive' and 'negative' classes.

6. evaluate_bert(trainer, tokenizer, val_texts, val_labels) Function:

This function is used to evaluate the performance of the BERT model on the validation dataset. It takes as inputs the trainer object, the tokenizer, the validation text data, and the corresponding labels.

It first tokenizes the validation data into the format that the BERT model can understand.
Then, it creates a PyTorch Dataset object from the tokenized data and the labels.
The function then uses the trainer to predict labels for the validation data. The model's predictions are converted from probabilities to binary class labels.
Lastly, it prints a classification report, which includes metrics such as precision, recall, and F1-score, providing a detailed evaluation of the model's performance.


7. train_and_evaluate_bert(data) Function:

This function is responsible for the training and evaluation of the BERT model. It takes as input the complete data.

It first prepares the data for the BERT model, including encoding the labels and tokenizing the text data.
It splits the data into training and validation sets.
It then creates PyTorch Dataset objects from the training and validation data.
It defines the training arguments using the TrainingArguments class from the Hugging Face library. These arguments specify various training parameters such as the number of epochs, batch size, and learning rate.
It checks if a pre-trained model exists. If it does, it loads that model; otherwise, it loads a new instance of the pretrained DistilBert model and trains it using the trainer object.
The trained model is then saved for future use.
Lastly, it calls the evaluate_bert function to evaluate the model's performance on the validation data.


9. flag_reviews_for_response(data) Function:

This function is used to flag reviews that require a response.

It loads the trained Logistic Regression model and the TF-IDF Vectorizer.
It defines the keywords that indicate a response is necessary and sets the sentiment score threshold for negative sentiment and the review length threshold.
It transforms the cleaned text data into TF-IDF vectors and calculates sentiment scores.
It then creates a new column 'response_flag' that indicates whether a response is needed based on the sentiment scores, review length, and presence of certain keywords.
The data is then divided into two datasets based on the 'response_flag' value and saved into separate CSV files.


10. is_response_needed(review) Function:

This function determines whether a response is needed for a given review. It works similarly to the flag_reviews_for_response function but operates on a single review.

11. predict_sentiment(review) Function:

This function predicts the sentiment of a review using all three models: Logistic Regression, Random Forest, and BERT. It prints out the predicted sentiment and the associated probability for each model.

12. main() Function:

This is the main function that ties all the above parts together. It prepares the data, trains and evaluates the models, flags reviews for response, and then enters a loop where it repeatedly asks the user to input a review and then predicts the sentiment of the input review.




## Instructions to Run the Code

1. Clone this repository to your local machine.
2. Install the necessary Python libraries mentioned in `requirements.txt`.
3. Run the main Python script.

## Results

The performance of the models is evaluated using metrics such as precision, recall, and F1-score. The detailed results are printed out in the form of classification reports. The models achieved the following accuracies on the test set:

- Logistic Regression: 89%
- Random Forest: 85%
- BERT: 87%


Model Performance:

Based on the results, it appears that all three models performed well on the dataset, with overall accuracies ranging from 85% to 89%. However, the BERT model achieved the highest precision for negative reviews (94%) and the Logistic Regression model achieved the highest F1-score for positive reviews (89%). This suggests that the BERT model might be more precise at identifying negative sentiments, while the Logistic Regression model might achieve a better balance between precision and recall for positive sentiments.


## Future Work

There is potential for further improvement of the models by tuning hyperparameters, increasing the size of the training data, or using more complex models.


