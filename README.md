# AI Powered Vocabulary Tutor (Group 26)

## Group Members
* [Elikem Asudo Gale-Zoyiku](mailto:elikem.gale-zoyiku@ashesi.edu.gh)
* [Pascal Okoli Mathias](mailto:pascal.mathias@ashesi.edu.gh)

## Project Description
Traditional educational approaches often struggle to captivate the attention of modern students immersed in digital technology. Inspired by our experiences as both student-tutors and learners, we recognize the need for a dynamic and interactive solution. The Virtual Vocabulary Tutor project addresses this gap by bridging conventional education with the digital age, catering to diverse learning styles and preferences. This application works on a K-Means clustering algorithm to gfroup words into clusters based on the words' lengths and automated readability indices. The word embeddings used are a Word2Vec model trained on a curated list of words. The K-Means algorithm is then trained on a subset of the list of words (1000 words). The program then randomly selects 4 words from the dataset, and pronounces one of them. The user is then asked to select the word that was pronounced. The program then analyses the user's response and provides feedback based on the user's performance. The program then selects another set of words and repeats the process.

## Project Objectives
The project's specific objectives are:
Develop an NLP-based vocabulary tutor capable of presenting interactive word options and questions.
Implement an error analysis mechanism to customise content and adapt to individual learning needs. Wrong answers the user chooses will be analysed to detect any trends.


## Project Deliverables
- A terminal based application that can be used to learn vocabulary.
- A README file that contains the project's documentation.
- A presentation that explains the project's implementation and results.
- A video that demonstrates the project's implementation and results.
- A trained model that can be used to predict the cluster of a word.

## Required Libraries
- Located in the requirements.txt file

## Installation Instructions
- Clone the repository
- Install the required libraries using the command `pip install -r requirements.txt`
- Run the program using the command `python main.py`

## Limitations
- The accent of the voice may not be so clear.
- The speed of the voice may be too fast for different users.


## Date Published
- 03-12-2023

## Demo
https://youtu.be/0Odlbi0jy8A 

## References
M. Kharis, Kisyani Laksono, and Suhartono. 2022. Utilization of NLP-Technology in Current Applications for Education and Research by Indonesian Student, Teacher, and Lecturer. 14th ed., Journal of Higher Education Theory &amp; Practice.

Holland, V.M., Sams, M.R., & Kaplan, J.D. (Eds.). (1996). Intelligent Language Tutors: Theory Shaping Technology (1st ed.). Routledge. https://doi.org/10.4324/9781315044811