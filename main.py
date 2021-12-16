import matplotlib.pyplot as plt #on laisse dans le doute
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

    binaryOutputs = []

    for i in range(len(outputs)):
        if outputs[i] == 'FAKE':
            binaryOutputs.append(0)
        else:
            binaryOutputs.append(1)

    data_labels = binaryOutputs
    x_train, x_test, y_train, y_test = train_test_split(data['text'], data_labels, test_size=0.1, random_state=0)

    #Methode TFIDF pour lire la réccurence des mots
    tfidf = TfidfVectorizer()

    tfidf_train = tfidf.fit_transform(x_train) 
    tfidf_test = tfidf.transform(x_test)
    tfidf_test_final = tfidf.transform(data['text'])

    # Classification -> Faut trouver un algo de classification, VOIR AVEC LE PROF
    # Ce documenter sur le PassiveAgressiveClassifier()
    # Description du site : Robust regression aims to fit a regression model 
    # in the presence of corrupt data: either outliers, 
    # or error in the model.
    passiveAgressiveClassifier = PassiveAggressiveClassifier()
    passiveAgressiveClassifier.fit(tfidf_train, y_train)
    y_pred = passiveAgressiveClassifier.predict(tfidf_test)

    #Score
    score = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {round(score * 100, 2)}%')

    #Matrice de confusion
    cmat = confusion_matrix(y_test,y_pred)
    print(cmat)