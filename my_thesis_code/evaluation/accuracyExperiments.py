import os
import numpy as np
import json
from glob import glob
import sys
sys.path.insert(0, '/Users/beatrizpaula/Desktop/Tese/my_thesis_code')



if __name__ == "__main__":

    test_dir = "/Volumes/DropSave/Tese/dataset/test_dictionary.json"
    '''with open(test_dir) as data:
        test_files ='''

    top = (1, 3, 5)

    predict_dir = "/Volumes/DropSave/Tese/trainedModels/firstTry"
    #for test in glob(predict_dir + "/*"):
        #seq_pred = test["seqs"]

    '''d = {}
    os.mkdir("/Volumes/DropSave/Tese/trainedModels/firstTry/test")
    with open("/Volumes/DropSave/Tese/trainedModels/firstTry/test/1.json", "w") as fp:
        json.dump(d, fp)'''

    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    print(g)





