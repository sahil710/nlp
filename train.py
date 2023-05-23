import nltk
import pickle5 as pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
# download required nltk data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

##Importing dataset--------------
import pandas as pd
from datasets import load_dataset

# Load the GoEmotions dataset
data = load_dataset('go_emotions', 'simplified')
train_dataset = pd.DataFrame(data['train'])
test_dataset = pd.DataFrame(data['test'])
val_dataset = pd.DataFrame(data['validation']) 


#Data Cleaning----------------------

#creating a column for new labels-----
# Define a function to group the labels
def group_labels(label):
    if label in [25,16,24]:
        return 'Sad & Grief & Remorse'
    elif label in [2,3]:
        return 'Anger & annoyance'
    elif label in [14,19]:
        return 'Fear & nervousness'
    elif label in [17,21]:
        return 'Joy & pride'
    elif label in [18,5]:
        return 'Love & caring'
    elif label in [26]:
        return 'Surprise'
    elif label in [0,4]:
        return 'Admiration & approval'
    elif label in [15,23]:
        return 'Gratitude & Relief'
    elif label in [6,7]:
        return 'Curiosity & Confusion'
    elif label in [11,10]:
        return 'Disgusting & Disapproval'
    elif label in [13,1]:
        return 'Excitement & amusement'
    elif label in [8,20]:
        return 'Optimistic & Desire'
    elif label in [12,9]:
        return 'Disappointment & embarrassment'
    elif label in [27]:
        return 'Neutral'

# Apply the group_labels function to the 'labels' column in all three datasets
train_dataset['new_label'] = train_dataset['labels'].apply(lambda x: group_labels(x[0]))
test_dataset['new_label'] = test_dataset['labels'].apply(lambda x: group_labels(x[0]))
val_dataset['new_label'] = val_dataset['labels'].apply(lambda x: group_labels(x[0]))
train_dataset.head()

#creating a column for numbering new labels-----
# Define a function to preprocess the labels
def preprocess_labels(labels):
    if isinstance(labels, (list, tuple, str)):
        label = labels[0]
    else:
        label = labels
        
    if label in [25,16,24]:
        return '1'
    elif label in [2,3]:
        return '2'
    elif label in [14,19]:
        return '3'
    elif label in [17,21]:
        return '4'
    elif label in [18,5]:
        return '5'
    elif label in [26]:
        return '6'
    elif label in [0,4]:
        return '7'
    elif label in [15,23]:
        return '8'
    elif label in [6,7]:
        return '9'
    elif label in [11,10]:
        return '10'
    elif label in [13,1]:
        return '11'
    elif label in [8,20]:
        return '12'
    elif label in [12,9]:
        return '13'
    elif label in [27]:
        return '14'

# Apply the preprocess_labels function to the train, test, and validation datasets
train_dataset['new_label_num'] = train_dataset['labels'].apply(preprocess_labels)
test_dataset['new_label_num'] = test_dataset['labels'].apply(preprocess_labels)
val_dataset['new_label_num'] = val_dataset['labels'].apply(preprocess_labels)


#Dropping unwanted columns id and labels:--------
### Droping unwanted columns
train_dataset = train_dataset.drop('id', axis=1)
train_dataset = train_dataset.drop('labels', axis=1)
test_dataset = test_dataset.drop('id', axis=1)
test_dataset = test_dataset.drop('labels', axis=1)
val_dataset = val_dataset.drop('id', axis=1)
val_dataset = val_dataset.drop('labels', axis=1)

##Removing columns with null values:-------
# Remove the row containing null value
train_dataset = train_dataset.dropna()
test_dataset = test_dataset.dropna()
val_dataset = val_dataset.dropna()

## Pre processing text--------
stop_words = [i for i in stopwords.words('english') if "n't" not in i and i not in ('not','no')]
tokenizers=[word_tokenize]
def process(tok):
    tok = tok.encode('ascii', 'ignore').decode('ascii')
    #Removing emojis
    tok = re.sub(r'[^\w\s]', '', tok)
    # tokenize words in text
    tok = word_tokenize(tok)
    #substitutes any white space
    tok = [re.sub('[^A-Za-z]+', '', word) for word in tok] 
    # lowercasing
    tok = [word.lower() for word in tok if word.isalpha()] 
    #Stopword
    tok = [word for word in tok if word not in stop_words]
    #Lemmtisation
    tok = [WordNetLemmatizer().lemmatize(word) for word in tok]
    #text = [stemmer.stem(word) for word in text] 
    tok = ' '.join(tok) 
    return tok
train_dataset['text_tokenized_no_punctuations'] = train_dataset['text'].apply(process) # this line applies process_text function to Sentence in dataset
test_dataset['text_tokenized_no_punctuations'] = test_dataset['text'].apply(process)
val_dataset['text_tokenized_no_punctuations'] = val_dataset['text'].apply(process)

##Creating logistic regression model:
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Define the pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=1.0,penalty='l1' ,solver='liblinear', max_iter=1000))
])

# Fit the pipeline to the training data
pipeline.fit(train_dataset['text_tokenized_no_punctuations'], train_dataset['new_label'])

# Make predictions on the test data
lr_exp3_p1 = pipeline.predict(test_dataset['text_tokenized_no_punctuations'])


# Evaluate the accuracy of the predictions
accuracy = accuracy_score(test_dataset['new_label'], lr_exp3_p1)
print('Accuracy:', accuracy)

pickle.dump(pipeline, open('flask-app/nlp_model.pkl', 'wb'))
