'''for _ in range(5):
    play_round_one()'''

from flask import Flask, render_template, request, redirect, url_for, flash
import random

app = Flask(__name__)

# ... (your existing code)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/play_round', methods=['POST','GET'])
def play_round():
    random_words = application_df['word'].sample(n=4).tolist()
    index = random.randint(0, 3)
    word_to_pronounce = random_words[index]

    play_audio(word_to_pronounce)
    time.sleep(0.5)

    return render_template('play_round.html', word_to_pronounce=word_to_pronounce, random_words=random_words, index=index)

@app.route('/handle_answer', methods=['POST'])
def handle_answer():
    user_input = int(request.form['user_input'])
    index = int(request.form['index'])
    word_to_pronounce = request.form['word_to_pronounce']

    if user_input == index + 1:
        flash("Correct!", 'success')
    else:
        cluster_of_incorrect_word = predict_cluster(word_to_pronounce)
        stored_clusters.append(cluster_of_incorrect_word)
        flash(f"Incorrect! Predicted cluster: {cluster_of_incorrect_word}", 'danger')

    return redirect(url_for('play_round'))

if __name__ == '__main__':
    app.run(debug=True)