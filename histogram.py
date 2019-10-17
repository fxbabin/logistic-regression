##################
#   LIBRAIRIES   #
##################

import pandas as pd
from argparse import ArgumentParser
import sys
import seaborn as sns
import numpy as np

##################
#   FUNCTIONS    #
##################

def get_arguments ():
    parser = ArgumentParser(description='Data generator program.')
    parser.add_argument('-f', '--file', help='csv file', required=True)
    res = parser.parse_args(sys.argv[1:])
    return (res.file)

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
        np.warnings.filterwarnings('ignore')
        df = df[['Hogwarts House', 'Arithmancy', 'Care of Magical Creatures']]
        for el in df.columns:
            if not isinstance(df[el].iloc[0], float) and el != "Hogwarts House":
                df = df.drop(el, axis=1)
            else:
                df[el] = df[el][df[el].map(lambda x: str(x) != "nan" or str(x) != "NaN")]
        sns.set(style="ticks", color_codes=True)
        sns_plot = sns.pairplot(df, hue="Hogwarts House", diag_kind="hist", plot_kws={'alpha':0.3})
        sns_plot.savefig("histogram.png")
    except Exception as e:
        print("Error while generating histogram ({})".format(e))
        sys.exit(-1)

if __name__ == "__main__":
    main()