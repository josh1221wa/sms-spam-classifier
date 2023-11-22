# SMS Spam Classifier

This code is an implementation of a SMS spam classifier using various machine learning models and techniques. It performs the following steps:

1. Data Preprocessing:
    - Reads the dataset from a CSV file and renames the columns.
    - Encodes the labels as numeric values.
    - Calculates the average word count in the messages.
    - Finds the number of unique words in the dataset.

2. Data Splitting:
    - Splits the dataset into training and testing sets.

3. Feature Extraction:
    - Uses TF-IDF vectorization to convert text messages into numerical features.

4. Model Training and Evaluation:
    - Trains a Multinomial Naive Bayes model as a baseline.
    - Evaluates the model using accuracy, precision, recall, and F1-score.
    - Trains three additional models using different architectures:
      - A model with a custom vectorization layer and embedding layer.
      - A model with bidirectional LSTM layers.
      - A model using transfer learning with Universal Sentence Encoder.
    - Evaluates the performance of each model.

5. Results Visualization:
    - Displays the evaluation results in a tabular format.
    - Plots the evaluation results for comparison.

The code is well-documented and organized into functions for better modularity and reusability. It utilizes popular machine learning libraries such as pandas, numpy, scikit-learn, matplotlib, and TensorFlow.

For more details, please refer to the code comments and the generated README file.

