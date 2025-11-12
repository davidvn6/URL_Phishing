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

    # Record when the model begins to run
    start = time.time()

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

