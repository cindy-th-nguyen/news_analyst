import matplotlib.pyplot as plt # On laisse dans le doute
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import pandas as pand
import matplotlib.pyplot as plt
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__, template_folder='website')

tfidf = TfidfVectorizer()
mlpClassifer = MLPClassifier()

@app.route('/')
def form_template():
    return render_template('index.html', result = '?')

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form.get("article")
    processed_text = [text]
    vecotrizedText = tfidf.transform(processed_text)
    predict = mlpClassifer.predict(vecotrizedText)
    predictProba = mlpClassifer.predict_proba(vecotrizedText)
    print(f'preditct : { predict }')
    print(f'preditct proba : { predictProba }')
    result = 'TRUE' if predict[0] != 0 else 'FAKE'
    return render_template('index.html', result=result)



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
            isFakeLabel.append(0)
        else:
            isFakeLabel.append(1)
    
    # On split le JDD
    train_article_inputs, test_articles, train_article_desired, test_article_desired = train_test_split(data['text'], isFakeLabel, test_size=0.5)

    # Methode TFIDF pour lire la réccurence des mots, fréquence. 
    # Vectorisation de la pensée
    # fréquence mot et leur réccurence.
    tfidf_train = tfidf.fit_transform(train_article_inputs) 
    tfidf_test = tfidf.transform(test_articles)

    # Classification Test -> MLP
    mlpClassifer.fit(tfidf_train, train_article_desired)
    prediction = mlpClassifer.predict(tfidf_test)

    # Data for KMEANS
    # SVD - Contribue à la précision du modèle 
    # Décomposition en valeurs singulières
    # Réduction de dimension
    svd = TruncatedSVD(2)
    data_fit = svd.fit_transform(tfidf_test)
    kmeans = KMeans(n_clusters= 2)
    label = kmeans.fit_predict(data_fit)
    print(data_fit.shape)
    print(label)

    # Score
    mlpClassifierScore = accuracy_score(test_article_desired, prediction)
    #score d'environ 90%
    print(f'mlpClassifier score: {mlpClassifierScore}')

    # Matrice de confusion
    # en general nous avons environ 10% de faux négatif / faux positif
    cmat = confusion_matrix(test_article_desired,prediction)
    print(cmat)
    display_cmat = ConfusionMatrixDisplay(confusion_matrix=cmat)
    display_cmat.plot()
    plt.show()

    # Kmeans plot
    supposedFakeLabel = data_fit[label == 0]
    supposedTrueLabel = data_fit[label == 1]
    plt.scatter(supposedFakeLabel[:,0] , supposedFakeLabel[:,1] , color = 'red')
    plt.scatter(supposedTrueLabel[:,0] , supposedTrueLabel[:,1] , color = 'blue')
    plt.show()

    app.run()