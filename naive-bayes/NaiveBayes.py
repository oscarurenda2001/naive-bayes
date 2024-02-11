import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict
'''
Primera versio del codi:
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import defaultdict

print('Lectura de la BD')
X = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv',sep=';')
print(X)
print('Comprovacio de nans')
print(X.isna().sum().sort_values() / len(X) * 100.)

print('Eliminació de nans')
X=X.dropna()
print(X.isna().sum().sort_values() / len(X) * 100.)

print('Creacio de train i test, separacio de les dades')
y = X['sentimentLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

print('Clasificació paraules (negatiu,positiu)')
frases = X['tweetText']
pos_neg = X['sentimentLabel']

cont_paraules_valor = defaultdict(lambda: [0, 0])
for tweetText, valor in zip(frases, pos_neg):
    palabras = tweetText.split()
    for palabra in palabras:
        cont_paraules_valor[palabra][valor] += 1
print('Algorisme de Bayes:')

probabilitats  = defaultdict(lambda: [0, 0])
tweets = X_train



tweetsNegatius = tweets[tweets['sentimentLabel'] == 0]
tweetsPositius = tweets[tweets['sentimentLabel'] == 1]
probNegatius = len(tweetsNegatius)/len(tweets)
probPositius = len(tweetsPositius)/len(tweets)
probs = [probNegatius,probPositius]
numParaules = [0,0]
for tweet in tweetsPositius['tweetText']:

    numParaules[1] += len(tweet.split())

for tweet in tweetsNegatius['tweetText']:
    numParaules[0] += len(tweet.split())

for paraula in cont_paraules_valor.keys():
    numPositiu = cont_paraules_valor[paraula][1]
    numNegatiu = cont_paraules_valor[paraula][0]
    auxPos = numPositiu/numParaules[1]
    auxNeg = numNegatiu/numParaules[0]

    if paraula not in probabilitats:
        probabilitats[paraula] = [0,0]
    probabilitats[paraula] = [auxPos,auxNeg]
print('Probabilitats fetes')

for tweet,valor in zip(tweets['tweetText'],tweets['sentimentLabel']):
    probabilitatActual = [1, 1]
    paraules = tweet.split()
    for paraula in paraules:
        if paraula in probabilitats:
            probabilitatActual[0] += probabilitats[paraula][0]
            probabilitatActual[1] += probabilitats[paraula][1]
        else:
            probabilitatActual[0] += probNegatius
            probabilitatActual[1] += probPositius
        probabilitatActual[0] += probs[0]
        probabilitatActual[1] += probs[1]
    if probabilitatActual[0] < probabilitatActual[1]:
        resultat = 1
    else:
        resultat = 0

    if resultat == 0 and resultat == valor:
        trueNeg += 1
    elif resultat == 1 and resultat == valor:
        truePos += 1
    elif resultat == 0 and resultat != valor:
        falseNeg += 1
    else:
        falsePos +=1

print("Accuracy:", (trueNeg + truePos)/(trueNeg+truePos+falseNeg + falsePos))
print("Precision:", truePos/(truePos+trueNeg))
print("Recall:", truePos/(truePos+falseNeg))
'''
print('Lectura de la BD')
X = pd.read_csv('FinalStemmedSentimentAnalysisDataset.csv', sep=';')
print(X)

print('Comprovacio de nans')
print(X.isna().sum().sort_values() / len(X) * 100.)

print('Eliminació de nans')
X = X.dropna()
print(X.isna().sum().sort_values() / len(X) * 100.)

print('Creacio de train i test, separacio de les dades')
y = X['sentimentLabel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
max_len = len(X_train)*0.2
print('Clasificació paraules (negatiu, positiu)')
frases = X['tweetText']
pos_neg = X['sentimentLabel']

cont_paraules_valor = defaultdict(lambda: [0, 0])

for tweetText, valor in zip(frases, pos_neg):
    palabras = tweetText.split()
    for palabra in palabras:
        if len(cont_paraules_valor) < max_len:
            cont_paraules_valor[palabra][valor] += 1

print('Algorisme de Bayes:')

probabilitats = defaultdict(lambda: [0, 0])
tweets = X_train

tweetsNegatius = tweets[tweets['sentimentLabel'] == 0]
tweetsPositius = tweets[tweets['sentimentLabel'] == 1]

probNegatius = len(tweetsNegatius) / len(tweets)
probPositius = len(tweetsPositius) / len(tweets)
probs = [probNegatius, probPositius]
numParaules = [0, 0]

for tweet in tweetsPositius['tweetText']:
    numParaules[1] += len(tweet.split())

for tweet in tweetsNegatius['tweetText']:
    numParaules[0] += len(tweet.split())

paraules = list(cont_paraules_valor.keys())
paraulesDesc = [1 / (numParaules[0] + len(paraules)), 1 / (numParaules[1] + len(paraules))]

for paraula in cont_paraules_valor.keys():
    numPositiu = cont_paraules_valor[paraula][1]
    numNegatiu = cont_paraules_valor[paraula][0]

    auxPos = (numPositiu + 1) / (numParaules[1] + len(paraules))
    auxNeg = (numNegatiu + 1) / (numParaules[0] + len(paraules))

    if paraula not in probabilitats:
        probabilitats[paraula] = [0, 0]
    probabilitats[paraula] = [auxNeg, auxPos]

print('Probabilitats fetes')

truePos = 0
trueNeg = 0
falsePos = 0
falseNeg = 0

for tweet, valor in zip(X_test['tweetText'], y_test):
    prob_pos = 1
    prob_neg = 1

    palabras = tweet.split()
    for palabra in palabras:
        if palabra in probabilitats:
            prob_pos *= probabilitats[palabra][1]
            prob_neg *= probabilitats[palabra][0]
        else:
            prob_pos *= paraulesDesc[1]
            prob_neg *= paraulesDesc[0]

    prob_pos *= probs[1]
    prob_neg *= probs[0]

    prediction = 1 if prob_pos > prob_neg else 0

    if prediction == valor:
        if valor == 1:
            truePos += 1
        else:
            trueNeg += 1
    else:
        if valor == 1:
            falseNeg += 1
        else:
            falsePos += 1

accuracy = (truePos + trueNeg) / (truePos + trueNeg + falsePos + falseNeg)
precision = truePos / (truePos + falsePos)
recall = truePos / (truePos + falseNeg)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)