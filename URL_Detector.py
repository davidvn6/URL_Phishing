# ===============================================================
# URL DETECTOR PROJECT
# ===============================================================

# Used to measure train time and run time
import time
# Used to process data
import pandas as pd
# Used to convert URLS into numerical features
from sklearn.feature_extraction.text import CountVectorizer
# Used to split data into train and test sets
from sklearn.model_selection import train_test_split
# Used for Naive Bayes
from sklearn.naive_bayes import MultinomialNB

# ===============================================================
# function trainModel() 
# used to train the Naive Bayes model 
# and print the evaluation metrics of said model
# ===============================================================

def trainModel():
    # Initialize the evaluation metric variables
    accuracy_number = None
    train_time = None
    run_time = None

    # Record when the function begins to run
    func_start = time.time()

    # Load the data
    df = pd.read_csv("Phishing_URL_Dataset.csv")

    # Look at first few rows to see how the data looks
    print(df.head())

    # Split data into the train and test sets
    # Input features is just the url of the given site
    x_data = df["URL"]
    # Target variable is 0 or 1, which indicates whether the URL is malicious or not
    y_data = df["label"]

    # Use train_test_split to split the data, we will have an 80% train and 20% test split
    x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data,
    test_size=0.2,          # 80/20 split
    random_state=42,        # Randomize the data
    stratify=y_data,             # Keep the class balance between train and test split
    )

    # Feature extraction
    # CountVectorizer will convert text into numbers for the ML model to learn from
    # analyzer="char_wb" to analyze the character substring instead of words
    # ngram_range=(3,5) to look at substrings that are 3 to 5 characters
    count_vector = CountVectorizer(analyzer="char_wb", ngram_range=(3,5))

    # Convert urls into numeric feature and then fit the data to the model
    train_features = count_vector.fit_transform(x_train)

    # Build a Multinominal Naive Bayes model because it works well with 
    # characters (integer features once we convert the char)
    MLmodel = MultinomialNB()
    
    # Record the time when the model training starts
    model_start = time.time()

    # Train the ML model
    MLmodel.fit(train_features, y_train)

    # Record time when model training ends
    model_end = time.time()

    # Calculate and print the training time
    train_time = model_end - model_start
    print(f"Model took {train_time} seconds to train")

    # Transform the test split so that the urls are converted to numeric features
    # and can be tested against the model
    test_features = count_vector.transform(x_test)
    