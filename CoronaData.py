import csv
import numpy as np
from sklearn.linear_model import LinearRegression

def normalize(max, min, val):
    return(val-min)/(max-min)

def transposeMatrix(array):
    numpyArray = np.array(array)
    transpose = numpyArray.T
    transposedArray = transpose.tolist()
    return transposedArray

def splitNormalize(dictionary, ratio, minList, maxList):
    limit = int(ratio * len(dictionary))
    trainingVals = []
    testVals = []
    count = 0
    for key in dictionary:
        valList = list(dictionary[key])
        normalList = [normalize(maxList[i], minList[i], valList[i]) for i in range(len(valList))]
        if count < limit:
            trainingVals.append(normalList)
        else:
            testVals.append(normalList)
        count+=1
    transposedTrainingVals = transposeMatrix(trainingVals)
    transposedTestVals = transposeMatrix(testVals)

    return trainingVals, testVals

"""
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

numpy_array = np.array(list_of_lists)
transpose = numpy_array.T
transpose `numpy_array`

transpose_list = transpose.tolist()

"""
def cost(listy, listx, m, b):
  predictedVals = [(m * x) + b for x in listx]
  sumCost = 0
  for i in range(len(predictedVals)):
    sumCost += (y[i] - predictedVals[i]) ** 2
  return sumCost/len(predictedVals)


def bGradient(listy, listx, m, b):
  predictedVals = [(m * x) + b for x in listx]
  bGradient = 0
  for i in range(len(predictedVals)):
    bGradient += ((listy[i] - predictedVals[i]) * -2)
  return bGradient/len(predictedVals)

def predict(features, weights, bias):
    array = np.dot(weights, features)
    predictions = [i + bias for i in array]
    return predictions


def mGradient(listy, listx, pos, weightList, b):
   predictedVals = [(weightList[pos] * x) + b for x in listx]
   mGradient = 0
   for i in range(len(predictedVals)):
    mGradient += ((listy[pos][i] - predictedVals[i]) * -2) * listx[i] 
   return mGradient/len(predictedVals)

def gradient(weightList, learningRate, features, labels, bias):
    for i in range(len(weightList)):
        weightList[i] -= mGradient(weightList, features, labels,i, bias) * learningRate
    return weightList


covidDict = {}
cases = []
factorIndicies = []
factors = ["total_tests_per_thousand","stringency_index", "population", "population_density", "aged_65_older", "gdp_per_capita", "hospital_beds_per_thousand", "life_expectancy"]
minFactors = [1000000000 for i in range(len(factors))]
maxFactors = [-1000000000 for i in range(len(factors))]
casePos = -1


with open("CoronaData2.csv", "r") as file:
    fileReader = csv.reader(file)
    index = 0
    for row in fileReader:
        for i in range(len(row)):
            if row[i] == "total_cases_per_million":
                casePos = i
            if row[i] == factors[index]:
                factorIndicies.append(i)
                index +=1
            if index == len(factors):
                break
        break

    for row in fileReader:
        countryName = row[2]
        if countryName == "location":
            continue
        if countryName not in covidDict:
            covidDict[countryName]  = []
            #for num in factorIndicies:
            for i in range(len(factorIndicies)):
                val = row[factorIndicies[i]]
                if len(val) == 0 or val is None:
                    covidDict.pop(countryName)
                    break
                else:
                    numVal = float(val)
                    covidDict[countryName].append(numVal)
                    cases.append(row[casePos])
                    if numVal > maxFactors[i]:
                        maxFactors[i] = numVal
                    if numVal < minFactors[i]:
                        minFactors[i] = numVal
        if len(covidDict) >= 213:
            break

trainingData, testData = splitNormalize(covidDict,0.8,minFactors, maxFactors)
trainingLabels = [cases[i] for i in range(0,len(trainingData))]
testLabels = [cases[i] for i in range(len(trainingData))]
print(len(trainingData), len(trainingLabels))
mlr = LinearRegression()
model = mlr.fit(trainingData, trainingLabels)
print(mlr.coef_)
npTrain = np.array(trainingData)
npTest = np.array(trainingLabels)
print("score = ", mlr.score(npTrain.astype(np.float64), npTest.astype(np.float64))







