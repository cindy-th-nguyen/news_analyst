import matplotlib.pyplot as plt # On laisse dans le doute
from sklearn.neural_network import MLPClassifier
import pandas as pand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, jsonify

@app.route('/api/fakenews/')
# Création de l'API
def resultFakeNews():
    dictionnary = {
        'isFake': 'false',
        'percentage': '{score}'
    }
    return jsonify(dictionnary)


if __name__ == "__main__":

    # Lecture du csv avec le framework pandas
    data = pand.read_csv('news.csv')
    print(f'data shape : {data.shape}')
    inputs = data['text']
    print(f'inputs shape : {inputs.shape}')
    outputs = data['label']
    print(f'label shape : {outputs.shape}')

    isFakeLabel = []
    for i in range(len(outputs)):
        if outputs[i] == 'FAKE':
            isFakeLabel.append(1)
        else:
            isFakeLabel.append(0)

    # On split le JDD
    train_article_inputs, test_articles, train_article_desired, test_article_desired = train_test_split(data['text'], isFakeLabel, test_size=0.1, random_state=0)

    # Methode TFIDF pour lire la réccurence des mots, fréquence. 
    # Vectorisation de la pensée
    # Voir les différents paramètres possible dans la DOC
    tfidf = TfidfVectorizer(max_features=21405)

    tfidf_train = tfidf.fit_transform(train_article_inputs) 
    tfidf_test = tfidf.transform(test_articles)

    # Classification -> Faut trouver un algo de classification, VOIR AVEC LE PROF
    mlpClassifer = MLPClassifier()
    mlpClassifer.fit(tfidf_train, train_article_desired)
    print(f'coef len: {len(mlpClassifer.coefs_[0])}')
    prediction = mlpClassifer.predict(tfidf_test)

    # Score
    score = accuracy_score(test_article_desired, prediction)
    # s = mlpClassifer.score(test_article_desired,prediction)
    print(f'score: {score}')

    # Matrice de confusion
    cmat = confusion_matrix(test_article_desired,prediction)
    print(cmat)

    