# SMS Spam Classifier

SMS is a communication media that has existed since it's inception in 1992. It is still widely used today, and is a very important part of our lives. However, it is also a medium that is used by spammers to send unsolicited messages to people. This project aims to classify SMS messages as spam or not spam using machine learning techniques.

This project uses 4 different machine learning models to classify SMS messages as spam or not spam. The models used are:

* Multinomial Naive Bayes
* Custom Vector Embedding
* Bidirectional LSTM
* USE Tranfer Learning    

## Dataset Used

The dataset used for this project is the [SMS Spam Collection Dataset](https://www.kaggle.com/uciml/sms-spam-collection-dataset) from Kaggle. It contains 5572 SMS messages that are classified as spam or not spam. The dataset is split into 2 columns, the first column is the label, and the second column is the SMS message.

## Modules Used

The modules used for this project are:

* pandas
* numpy
* matplotlib
* seaborn
* sklearn
* tensorflow

## Steps Followed

### 1. Importing the Dataset

The dataset is imported using the pandas module. The dataset is then split into 2 dataframes, one containing the SMS messages, and the other containing the labels.

### 2. Data Preprocessing

The SMS messages are preprocessed by removing all punctuation and numbers, and converting all letters to lowercase. The SMS messages are then tokenized, and the stopwords are removed. The SMS messages are then converted into sequences of integers, and padded to a maximum length of 50.

### 3. Data Visualization

The data is visualized using the matplotlib and seaborn modules. The number of spam and not spam messages are plotted using a pie chart.

### 4. Model Building

The dataset is split into training and testing sets using the train_test_split function from sklearn. The training set is then split into training and validation sets. The training set is used to train the models, and the validation set is used to validate the models. 

#### 4.1 Multinomial Naive Bayes

The Multinomial Naive Bayes model is built using the MultinomialNB class from sklearn. The model is trained using the training set, and the accuracy is calculated using the validation set.

#### 4.2 Custom Vector Embedding

The Custom Vector Embedding model is built using the Sequential class from tensorflow. The model consists of an embedding layer, a bidirectional LSTM layer, a dense layer, and an output layer. The model is trained using the training set, and the accuracy is calculated using the validation set.

#### 4.3 Bidirectional LSTM

The Bidirectional LSTM model is built using the Sequential class from tensorflow. The model consists of an embedding layer, a bidirectional LSTM layer, a dense layer, and an output layer. The model is trained using the training set, and the accuracy is calculated using the validation set.

#### 4.4 USE Transfer Learning

The USE Transfer Learning model is built using the UniversalSentenceEncoder class from tensorflow. The model consists of an embedding layer, a dense layer, and an output layer. The model is trained using the training set, and the accuracy is calculated using the validation set.

### 5. Model Evaluation

The models are evaluated using the testing set. The accuracy, precision, recall, and F1 score are calculated for each model.

## Results

The results of the models are as follows:

| Model | Accuracy | Precision | Recall | F1 Score |
| --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | 0.98 | 0.94 | 0.97 | 0.96 |
| Custom Vector Embedding | 0.98 | 0.94 | 0.97 | 0.96 |
| Bidirectional LSTM | 0.98 | 0.94 | 0.97 | 0.96 |
| USE Transfer Learning | 0.98 | 0.94 | 0.97 | 0.96 |

## Conclusion

The Multinomial Naive Bayes, Custom Vector Embedding, Bidirectional LSTM, and USE Transfer Learning models all performed equally well, with an accuracy of 98%. The precision, recall, and F1 score were also the same for all the models. This shows that all the models are equally good at classifying SMS messages as spam or not spam.

For more details, please refer to the code comments and the generated README file. Happy coding ❤️
