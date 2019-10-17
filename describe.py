##################
#   LIBRAIRIES   #
##################

import pandas as pd
from argparse import ArgumentParser
import sys
import numpy as np
from math import floor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

##################
#   FUNCTIONS    #
##################

def get_arguments ():
    parser = ArgumentParser(description='Data generator program.')
    parser.add_argument('-f', '--file', help='csv file', required=True)
    res = parser.parse_args(sys.argv[1:])
    return (res.file)

def get_quartile(arr, count, nb=1):
    index = (count - 1) * (nb * 0.25)
    if floor(index) == index :
        return arr[int(index)]
    return (arr[floor(index)] + (arr[floor(index) + 1] - arr[floor(index)]) * (index - floor(index)))

def skewness(arr, mean, std):
    size = len(arr)
    coeff = size / ((size - 1) * (size - 2))
    skewness = 0
    for e in arr:
        skewness += ((e - mean) / std) ** 3
    return (coeff * skewness)

def get_min_max(arr):
    arr_count = 0
    arr_sum = 0.0
    arr_min = 1.0e9
    arr_max = -1.0e9

    for el in arr:
        arr_min = el if el < arr_min else arr_min
        arr_max = el if el > arr_max else arr_max
        arr_count += 1
        arr_sum += el
    arr_mean = arr_sum / arr_count
    return (arr_min, arr_max, arr_count, arr_mean)

def describe(df):
    desc = {}
    
    for e in ['name', 'count', 'mean', 'min', 'max', 'std', '25%', '50%', '75%', 'skewness']:
        desc[e] = []
    
    for key in df.keys():
        arr = sorted(df[key][df[key].map(lambda x: str(x) != "nan")])
        arr_min, arr_max, arr_count, arr_mean = get_min_max(arr)
        
        # variance
        var = 0.0
        for el in arr:
            var += ((el - arr_mean) ** 2)
        var /= (arr_count - 1)
        arr_std = np.sqrt(var)

        desc['name'].append(key)
        desc['count'].append(float(arr_count))
        desc['mean'].append(arr_mean)
        desc['min'].append(arr_min)
        desc['max'].append(arr_max)
        desc['std'].append(arr_std)
        desc['25%'].append(get_quartile(arr, arr_count, 1))
        desc['50%'].append(get_quartile(arr, arr_count, 2))
        desc['75%'].append(get_quartile(arr, arr_count, 3))
        desc['skewness'].append(skewness(arr, arr_mean, arr_std))
    return desc

def print_description(desc):
    padding = []
    for _ in desc['count']:
        padding.append(0)
    
    for key in desc.keys():
        for idx in range(0, len(padding)):
            str_len = len(str(desc[key][idx]))
            num_len = len(str(desc[key][idx]).split('.')[0])
            tmp = str_len if str_len == num_len else (num_len + 9)
            if (tmp + 2) > padding[idx]:
                padding[idx] = tmp + 2

    for col in ['name', 'count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness']:
        out = ""
        out += "{:8}".format("") if col == "name" else "{:8}".format(col)
        
        for idx, el in enumerate(desc[col]):
            if isinstance(el, float):
                out += ("{:.6f}".format(el)).rjust(padding[idx])
            else:
                out += str(el).rjust(padding[idx])
        print(out)

def pca(df):
    x = df
    x = x.drop("Hogwarts House", axis=1)

    y = df["Hogwarts House"]
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents
        , columns = ['pc1', 'pc2'])

    finalDf = pd.concat([principalDf, df['Hogwarts House']], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC 1', fontsize = 15)
    ax.set_ylabel('PC 2', fontsize = 15)
    ax.set_title('PCA analysis of dataset', fontsize = 20)
    targets = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    colors = ['r', 'g', 'b', 'y']

    for target, color in zip(targets,colors):
        indicesToKeep = finalDf['Hogwarts House'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'pc1']
                , finalDf.loc[indicesToKeep, 'pc2']
                , c = color
                , s = 50)
    ax.legend(targets)
    plt.savefig("pca.png")

def main():
    try:
        filename = get_arguments()
    except Exception as e:
        print("Error while getting arguments ({})".format(e))
        sys.exit(-1)
    
    try:
        df = pd.read_csv(filename)
    except Exception as e:
        print("Error while reading csv file ({})".format(e))
        sys.exit(-1)

    try:
        df = df.fillna(df.median())
        for el in df.columns:
            if el == "Hogwarts House":
                continue
            if not isinstance(df[el].iloc[0], float):
                df = df.drop(el, axis=1)
        pca(df)
    except Exception as e:
        print("Error while performing pca ({})".format(e))
        sys.exit(-1)

    try:
        # removing non float columns
        df = pd.read_csv(filename)
        for el in df.columns:
            if not isinstance(df[el].iloc[0], float):
                df = df.drop(el, axis=1)
    except Exception as e:
        print("Error while removing columns from dataframe ({})".format(e))
        sys.exit(-1)
    
    try:
        description = describe(df)
        print_description(description)
    except Exception as e:
        print("Error in describe function ({})".format(e))

if __name__ == "__main__":
    main()