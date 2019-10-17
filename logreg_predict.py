##################
#   LIBRAIRIES   #
##################

import pandas as pd
from argparse import ArgumentParser
import sys
import numpy as np

##################
#   FUNCTIONS    #
##################

def transform(arr, house):
    return [((elem == house) * 1.0) for elem in arr]

def transform_round(arr):
    return [round(elem) for elem in arr]

def precision(pred, ground_truth):
    true_pos = 0
    false_pos = 0
    for tmp, gt in zip(pred, ground_truth):
        if tmp == 1 and gt == 1:
            true_pos += 1
        elif tmp == 1 and gt == 0:
            false_pos += 1
    return((true_pos / (true_pos + false_pos + 0.001)))

def recall(pred, ground_truth):
    true_pos = 0
    false_neg = 0
    for tmp, gt in zip(pred, ground_truth):
        if tmp == 1 and gt == 1:
            true_pos += 1
        elif tmp == 0 and gt == 1:
            false_neg += 1
    return((true_pos / (true_pos + false_neg + 0.001)))

def f1_score(pred, ground_truth):
    prec = precision(pred, ground_truth)
    rec = recall(pred, ground_truth)
    return (2 * ((prec * rec) / (prec + rec + 0.001)))

def all_houses_score(func, X, y_true, dic):
    mean = 0.0
    for house in ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]:
        y_gt = transform(y_true, house)
        y_pred = transform_round(predict(X, dic[house]))
        mean += func(y_pred, y_gt)
    return (mean / 4)

def get_arguments ():
    parser = ArgumentParser(description='Data generator program.')
    parser.add_argument('-f', '--file', help='csv file', required=True)
    parser.add_argument('-t', '--true', help='csv file', required=True)
    res = parser.parse_args(sys.argv[1:])
    return (res.file)

def weights_to_dict(df):
    dic = {}
    for house in df['house'].values:
        dic[house] = np.array([float(e) for e in df[df["house"] == house]['weights'].values[0].split(';')])
    return dic

def featureNormalize(X):
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    X = (X - x_min)/ (x_max - x_min)
    return X, x_min, x_max

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))

def get_arguments():
	parser = ArgumentParser(description='Data generator program.')
	parser.add_argument('-f', '--file', help='mileage', required=True)
	res = parser.parse_args(sys.argv[1:])
	return (res.file)

def predict_all(X, dic):
    
    rav = predict(X, dic["Ravenclaw"])
    sly = predict(X, dic["Slytherin"])
    gry = predict(X, dic["Gryffindor"])
    huf = predict(X, dic["Hufflepuff"])
    
    pred = []
    arr = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    for idx in range(len(rav)):
        i = 0
        maxx = 0.0
        max_idx = 0
        for e in [rav[idx], sly[idx], gry[idx], huf[idx]]:
            if e > maxx:
                maxx = e
                max_idx = i
            i += 1
        pred.append(arr[max_idx])
#         pred.append(max([rav[idx], sly[idx], gry[idx], huf[idx]]))
    return pred


def main():
    try:
        filename = get_arguments()
        df = pd.read_csv(filename)

        y_true = list(df["Hogwarts House"])
        
        df = df.drop("Hogwarts House", axis=1)
        df = df.fillna(df.median())
        for el in df.columns:
            if not isinstance(df[el].iloc[0], float):
                df = df.drop(el, axis=1)
        x_norm = pd.DataFrame(featureNormalize(df)).values[0,0]
        weights = pd.read_csv('Models/weights.csv')
        weights_dic = weights_to_dict(weights)

        predictions = predict_all(x_norm, weights_dic)

        with open("houses.csv", "w+") as house_file:
            house_file.write("Index,Hogwarts House\n")
            for idx, pred in enumerate(predictions):
                house_file.write("{},{}\n".format(idx,pred))
    except Exception as e:
        print("Error while training dataset ({})".format(e))
        sys.exit(-1) 

if __name__ == "__main__":
    main()