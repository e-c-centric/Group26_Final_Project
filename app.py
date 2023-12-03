import tkinter as tk
from tkinter import messagebox
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
from termcolor import colored

def print_colored(text, color="white"):
    print(colored(text, color))

def display_intro():
    print_colored("Welcome to the KooKooKaaKaa Vocab Game!", "cyan")
    print_colored("Listen to the word and choose the correct option.", "cyan")
    print()

def display_result(is_correct, word=None):
    if is_correct:
        print_colored("Correct!!", "green")
    else:
        print_colored(f"Incorrect! The word was {word}", "red")

def display_score(score):
    print_colored(f"Score: {score}", "cyan")


def play_audio(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 100)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()


kmeans = joblib.load(r'Group26_Final_Project/kmeans_model.pkl')

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

    display_intro()

    print_colored(f"Listen to the word carefully: ", "green")
    play_audio(word_to_pronounce)
    time.sleep(0.5)

    print_colored("\nSelect the correct option by index:", "cyan")
    for i, word in enumerate(random_words):
        print_colored(f"{i + 1}. {word}", "yellow")

    user_input = int(input("Enter the index of the correct option: "))

    if user_input == index + 1:
        display_result(True)
    else:
        cluster_of_incorrect_word = predict_cluster(word_to_pronounce)
        stored_clusters.append(cluster_of_incorrect_word)
        display_result(False, word_to_pronounce)

    score = 100 - (len(stored_clusters) * 20)
    display_score(score)

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
    print(f"Score: {score}")


class WordGameGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Word Clustering Game")
        self.score = 0
        self.stored_clusters = []
        self.round_number = 0

        self.random_words = []
        self.word_to_pronounce = ""

        self.instruction_label = tk.Label(master, text="")
        self.instruction_label.pack()

        self.word_label = tk.Label(master, text="")
        self.word_label.pack()

        self.option_buttons = []
        for i in range(4):
            button = tk.Button(master, text="", command=lambda idx=i: self.check_answer(idx + 1))
            button.pack()
            self.option_buttons.append(button)

        self.score_label = tk.Label(master, text="Score: 0")
        self.score_label.pack()

        self.play_round_one()

    def play_round_one(self):
        self.round_number += 1
        random_words = application_df['word'].sample(n=4)
        self.random_words = random_words.tolist()
        index = random.randint(0, 3)
        self.word_to_pronounce = self.random_words[index]

        self.instruction_label.config(text=f"Round {self.round_number}: Listen to the word carefully:")
        self.word_label.config(text=self.word_to_pronounce)
        play_audio(self.word_to_pronounce)
        time.sleep(0.1)

        for i, word in enumerate(self.random_words):
            self.option_buttons[i].config(text=f"{i + 1}. {word}")

    def play_round(self):
        random_words = application_df['word'].sample(n=4)
        random_words = random_words.tolist()

        if self.stored_clusters:
            filtered_words = application_df[application_df['cluster'].isin(self.stored_clusters)]['word'].sample(n=3)
            self.word_to_pronounce = random.choice([self.word_to_pronounce] + filtered_words.tolist())
        else:
            self.word_to_pronounce = random_words[random.randint(0, 3)]

        self.instruction_label.config(text="Listen to the word:")
        self.word_label.config(text=self.word_to_pronounce)
        play_audio(self.word_to_pronounce)
        time.sleep(0.1)

        for i, word in enumerate(random_words):
            self.option_buttons[i].config(text=f"{i + 1}. {word}")

    def check_answer(self, user_input):
        if user_input == self.random_words.index(self.word_to_pronounce) + 1:
            messagebox.showinfo("Correct!", "Great job! You got it right.")
            self.score += 20
            self.stored_clusters = []
            self.score_label.config(text=f"Score: {self.score}")

            if self.round_number < 5:
                self.play_round_one()
            elif self.round_number < 10:
                self.play_round()
            else:
                messagebox.showinfo("Game Over", f"Game Over! Your final score is {self.score}")
                self.master.destroy()

        else:
            cluster_of_incorrect_word = predict_cluster(self.word_to_pronounce)
            messagebox.showerror("Incorrect", f"Incorrect! The word was {self.word_to_pronounce}")
            self.stored_clusters.append(cluster_of_incorrect_word)
            self.score = max(0, self.score - 20)
            self.score_label.config(text=f"Score: {self.score}")

root = tk.Tk()
app = WordGameGUI(root)

root.mainloop()