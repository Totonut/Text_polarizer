import glob
from ml_toolkit import *

NB_KEYWORDS = 2
NB_FEATURES = 16 + 2 * NB_KEYWORDS

if __name__ == "__main__":
    left_files = glob.glob("features/*G")
    right_files = glob.glob("features/*D")
    inputs, outputs, lefts, rights = [], [], [], []
    for f in left_files:
        fi = open(f, "r")
        res = []
        for i in range(NB_FEATURES):
            res.append(float(fi.readline()))
        lefts.append(res)
    for f in right_files:
        fi = open(f, "r")
        res = []
        for i in range(NB_FEATURES):
            res.append(float(fi.readline()))
        rights.append(res)
    i = 0
    while i < len(left_files) and i < len(right_files):
        inputs.append(lefts[i])
        outputs.append([0])
        inputs.append(rights[i])
        outputs.append([1])
        i += 1
    while i < len(left_files):
        inputs.append(lefts[i])
        outputs.append([0])
        i += 1
    while i < len(right_files):
        inputs.append(rights[i])
        outputs.append([1])
        i += 1
    neural_net = NeuralNetwork([NB_FEATURES, 1], activation=TanH)
    print(inputs)
    print(outputs)
    neural_net.train(inputs, outputs)
    print(neural_net.predict(inputs[0]))
    print(neural_net.predict(inputs[1]))
    print(neural_net.predict(inputs[2]))
