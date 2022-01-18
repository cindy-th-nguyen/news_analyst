import matplotlib.pyplot as plt #on laisse dans le doute
from sklearn.neural_network import MLPClassifier
import pandas as pand
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

if __name__ == "__main__":

    #lecture du csv avec le framework pandas
    data = pand.read_csv('news.csv')
    print(f'data shape : {data.shape}')
    inputs = data['text']
    print(f'inputs shape : {inputs.shape}')
    outputs = data['label']
    print(f'label shape : {outputs.shape}')

    isFakeLabels = []
    for i in range(len(outputs)):
        if outputs[i] == 'FAKE':
            isFakeLabels.append(0)
        else:
            isFakeLabels.append(1)

    #On split le JDD
    train_article_inputs, test_articles, train_article_desired, test_article_desired = train_test_split(data['text'], isFakeLabels, test_size=0.1, random_state=0)

    #Methode TFIDF pour lire la réccurence des mots, fréquence.
    #Vectorisation du text
    tfidf = TfidfVectorizer()

    tfidf_train = tfidf.fit_transform(train_article_inputs) 
    tfidf_test = tfidf.transform(test_articles)
    tfidf_test_final = tfidf.transform(data['text'])

    # Classification -> Faut trouver un algo de classification, VOIR AVEC LE PROF
    # Trouver un Classifier qui convient
    mlpClassifer = MLPClassifier()
    mlpClassifer.fit(tfidf_train, train_article_desired)
    prediction = mlpClassifer.predict(tfidf_test)

    #Score
    score = accuracy_score(test_article_desired, prediction)
    print(f'score: {round(score * 100, 2)}%')

    #Matrice de confusion
    cmat = confusion_matrix(test_article_desired,prediction)
    print(cmat)