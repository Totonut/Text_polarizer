import glob
from features_extractor import FeaturesExtractor
from keywords import left_vs_right_keywords
import nltk.tag.stanford as st

def extract_component(f, keywords):
    # Apply each function
    name = f.name
    fe = FeaturesExtractor(f, keywords)
    data = fe.extract_features()

    # Write to file
    new_file = open("features/" + name[6:], "w")
    new_file.write('\n'.join([str(e) for e in data]))

def extract_keywords(tagger):
    left, right = left_vs_right_keywords(tagger)
    left = [e[0] for e in left if len(e[0]) > 3]
    right = [e[0] for e in right if len(e[0]) > 3]
    left = left[:50]
    right = right[:50]
    return list(set().union(left, right))

if __name__ == "__main__":
    tagger = st.StanfordPOSTagger("stanford-postagger-full-2014-08-27/models/french.tagger", "stanford-postagger-full-2014-08-27/stanford-postagger.jar")
    files = glob.glob("train/*")
    keywords = extract_keywords(tagger)
    print(keywords)
    for i in range(len(files)):
        print("{} / {}...".format(i + 1, len(files)))
        extract_component(open(files[i], "r"), keywords)
