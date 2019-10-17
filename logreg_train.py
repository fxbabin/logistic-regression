##################
#   LIBRAIRIES   #
##################

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt

##################
#   FUNCTIONS    #
##################

def sigmoid(z):
    return (1 / (1 + np.exp(-z)))

def predict(X, theta):
      return(sigmoid(np.dot(X, theta)))
    
def cost(X, y, theta):
    return((-1 / X.shape[0]) * np.sum(y * np.log(predict(X, theta)) + (1 - y) * np.log(1 - predict(X, theta))))

def featureNormalize(X):
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    X = (X - x_min)/ (x_max - x_min)
    return X, x_min, x_max

def mini_batch(X, y, batch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, X.shape[0] - batch_size + 1, batch_size):
        idx_batch = indices[start_idx:start_idx + batch_size]
        yield X[idx_batch], y[idx_batch]

def exponentionnal_decay(alpha_0, epochs, decay_rate):
    return(alpha_0 * np.exp(-decay_rate * epochs))

def logreg_fit(X, y, theta, alpha, num_iters, 
        batch_size=-1,
        decay_rate=0.0001,
        tol=0.000001):
    
    # Initialiser certaines variables utiles
    m = X.shape[0]
    J_history = []
    alpha_0 = alpha
    b_size = m if (batch_size <= 0 or batch_size > m) else batch_size
    decay_rate = 0.0 if b_size == m else decay_rate
    prev_cost = 0
    
    for i in tqdm(range(num_iters)):
        
        for batch in mini_batch(np.array(X), y, b_size):
            X_tmp, y_tmp = batch
            diff = np.dot((predict(X_tmp,theta) - y_tmp), X_tmp)
            theta = theta - alpha * (diff / m)
        
        # tol
        curr_cost = cost(X, y, theta)
        if abs(prev_cost - curr_cost) < tol:
            print("training finished in {} iterations".format(i))
            return theta, J_history
        prev_cost = curr_cost

        # learning rate decay
        alpha = exponentionnal_decay(alpha_0, i, decay_rate)
        J_history.append(cost(X, y, theta))
    return theta, J_history

def save_weights(weights):
    def print_w(name, theta):
        out = name + ","
        for e in theta:
            out += "{:.6f};".format(e)
        return out[:-1] + "\n"
    
    with open("Models/weights.csv", "w+") as weight_file:
        weight_file.write("house,weights\n")
        for house, weight in weights.items():
            weight_file.write(print_w (house, weight))

def show_history(J_history):
    plt.clf()
    fig = plt.figure()
    ax = plt.axes()
    for house, history in J_history.items():
        ax.plot(history, label=house)
    plt.xlabel("iterations")
    plt.ylabel("cost")
    plt.title("Cost evolution during training")
    ax.legend()
    plt.savefig("cost_history.png")
    plt.clf()


def training(df, alpha=0.05, iters=250, batch_size=-1, decay=0.0, tol=0.000001):
    df = df.fillna(df.median())

    for el in df.columns:
        if el == "Hogwarts House":
            continue
        if not isinstance(df[el].iloc[0], float):
            df = df.drop(el, axis=1)

    weights = {}
    J_history = {}
    for house in ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]:
        x = df.drop("Hogwarts House", axis=1)
        x = pd.DataFrame(featureNormalize(x)).values[0,0]
        y = (df["Hogwarts House"] == house) * 1 
        
        print("Training on {} house".format(house))
        weights[house] = np.zeros(x.shape[1], dtype=float)
        weights[house], J_history[house] = logreg_fit(
            np.array(x), y, weights[house], alpha, iters,
            batch_size=batch_size, decay_rate=decay, tol=tol)
    save_weights(weights)
    show_history(J_history)

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

def weights_to_dict(df):
    dic = {}
    for house in df['house'].values:
        dic[house] = np.array([float(e) for e in df[df["house"] == house]['weights'].values[0].split(';')])
    return dic

def main():
    try:
        df = pd.read_csv("Data/dataset_train.csv")
    except Exception as e:
        print("Error while reading csv file ({})".format(e))
        sys.exit(-1)
    try:
        training(df, 0.1, 10000, tol=0.000005, decay=0.0001)
    except Exception as e:
        print("Error while training dataset ({})".format(e))
        sys.exit(-1)

    try:
        df = pd.read_csv('Data/dataset_train.csv')

        y_true = list(df["Hogwarts House"])
        df = df.fillna(df.median())
        for el in df.columns:
            if el == "Hogwarts House":
                continue
            if not isinstance(df[el].iloc[0], float):
                df = df.drop(el, axis=1)
        x_test = df.drop("Hogwarts House", axis=1)
        x_norm = pd.DataFrame(featureNormalize(x_test)).values[0,0]

        weights = pd.read_csv('Models/weights.csv')
        weights_dic = weights_to_dict(weights)
        print("accuracy on training set: {}".format(all_houses_score(precision, x_norm, y_true, weights_dic)))
        print("recall on training set: {}".format(all_houses_score(recall, x_norm, y_true, weights_dic)))
        print("f1_score on training set: {}".format(all_houses_score(f1_score, x_norm, y_true, weights_dic)))
    except Exception as e:
        print("Error while training dataset ({})".format(e))
        sys.exit(-1) 

if __name__ == "__main__":
    main()