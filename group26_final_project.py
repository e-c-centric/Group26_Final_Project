import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
import random
import textstat
import langid
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import time
import joblib
import pyttsx3


def play_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()


kmeans = joblib.load('Group26_Final_Project\kmeans_model.pkl')

with open('Group26_Final_Project\words.txt', 'r') as file:
    words = file.read().split()

refined_words = [word for word in words if word.isalpha() and len(word) >= 3]

with open('application_words.txt', 'w') as file:
    file.write('\n'.join(refined_words))

random_sample = random.sample(refined_words, 1000)
with open('training_words.txt', 'w') as file:
    file.write('\n'.join(random_sample))

file_path = 'training_words.txt'
training_df = pd.read_csv(file_path, sep= " ", names=['word'])
training_df = training_df.dropna()

file_path = 'application_words.txt'
application_df = pd.read_csv(file_path, sep= " ", names=['word'])
application_df = application_df.dropna()

from gensim.models import Word2Vec
word2vec_model = Word2Vec(sentences=[application_df['word'].tolist()], vector_size=100, window=5, min_count=1, workers=4)
def word_to_vec(word):
    try:
        return word2vec_model.wv[word]
    except KeyError:
        return None


def extract_features(word):
    try:
        syllables = textstat.syllable_count(word)
        ari = textstat.automated_readability_index(word)
        length = len(word)
        #language, _ = langid.classify(word)

        return {'syllables': syllables, 'ari': ari, 'length': length, 'language': None}
    except Exception as e:
        print(f"Error extracting features for '{word}': {e}")
        return None

def predict_cluster(word):
    # Extract features for the word (similar to the preprocessing steps during training)
    word_vector = word_to_vec(word)
    features = extract_features(word)
    ari = features['ari']
    length = features['length']
    #lang = features['language']
    features=[ari,length]
    if None in features or word_vector is None:
        print(f"Error extracting features for '{word}'. Unable to predict the cluster.")
        return None
    input_features = np.concatenate([word_vector, features])
    input_features = input_features.reshape(1, -1)
    predicted_cluster = kmeans.predict(input_features)
    return predicted_cluster[0]


stored_clusters = []
score = 0
def play_round_one():
    global stored_clusters
    global score
    random_words = application_df['word'].sample(n=4)
    random_words = random_words.tolist()
    index = random.randint(0, 3)
    word_to_pronounce = random_words[index]

    print(f"\nListen to the word carefully: {word_to_pronounce}")
    play_audio(word_to_pronounce)
    time.sleep(0.5)

    print("\nSelect the correct option by index:")
    for i, word in enumerate(random_words):
        print(f"{i + 1}. {word}")

    user_input = int(input("Enter the index of the correct option: "))
    if user_input == index + 1:
        print("Correct!!")
    elif user_input == index + 1:
        cluster_of_incorrect_word = predict_cluster(word_to_pronounce)
        stored_clusters.append(cluster_of_incorrect_word)

    score = 100 - (len(stored_clusters) * 20)
    print(f"Score: {score}")

def play_round():
    global stored_clusters
    global score
    random_words = application_df['word'].sample(n=4)
    random_words = random_words.tolist()
    if stored_clusters:
        filtered_words = application_df[application_df['cluster'].isin(stored_clusters)]['word'].sample(n=3)
        word_to_pronounce = random.choice([word_to_pronounce] + filtered_words.tolist())
    else:
        word_to_pronounce = random_words[random.randint(0, 3)]

    print(f"\nListen to the word: {word_to_pronounce}")
    play_audio(word_to_pronounce)
    time.sleep(0.5)

    print("\nSelect the correct option by index:")
    for i, word in enumerate(random_words):
        print(f"{i + 1}. {word}")

    user_input = int(input("Enter the index of the correct option: "))

    print(f"User input: {user_input}")

    if user_input == random_words.index(word_to_pronounce) + 1:
        print("Correct!!")
        stored_clusters = []
    else:
        cluster_of_incorrect_word = predict_cluster(word_to_pronounce)
        print(f"Incorrect! The word was {word_to_pronounce}")
        stored_clusters.append(cluster_of_incorrect_word)
    score = score + (100 - (len(stored_clusters) * 20))

for _ in range(5):
    play_round_one()
for _ in range(5):
    play_round()

print("\nStored clusters:", stored_clusters)
print("Use these clusters to regulate the randomization in future iterations.")
