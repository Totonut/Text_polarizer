import glob

NB_KEYWORDS = 2
NB_FEATURES = 16 + 2 * NB_KEYWORDS

if __name__ == "__main__":
    left_files = glob.glob("features/*G")
    right_files = glob.glob("features/*D")
    left_res = [0.] * NB_FEATURES
    right_res = [0.] * NB_FEATURES
    for f in left_files:
        fi = open(f, "r")
        for i in range(NB_FEATURES):
            left_res[i] += float(fi.readline())
    for f in right_files:
        fi = open(f, "r")
        for i in range(NB_FEATURES):
            right_res[i] += float(fi.readline())
    left_res = [e / len(left_files) for e in left_res]
    right_res = [e / len(right_files) for e in right_res]
    for i in range(NB_FEATURES):
        print("LEFT: " + str(left_res[i]) + "\nRIGHT: " + str(right_res[i]))
