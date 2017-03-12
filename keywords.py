import nltk
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import word_tokenize
from collections import defaultdict
import glob

TYPES = [
    "PUNC",
    "DET",
    "PROREL",
    "P",
    "CLS",
    "CC",
    "CLO",
    "CS",
    "P",
]

def get_most_used_words(words):
    t = words.items()
    return sorted(t, key=lambda t: t[1], reverse=True)

def get_words_of_files(regex, tagger):
    stemmer = FrenchStemmer()
    res = defaultdict(lambda: 0)
    files = glob.glob(regex)
    for f in files:
        fi = open(f, "r")
        s = fi.read()
        tokens = word_tokenize(s)
        tags = tagger.tag(tokens)
        for w in [stemmer.stem(tokens[i].lower()) for i in range(len(tokens)) if tags[i] not in TYPES]:
            res[w] += 1
    return res

def left_vs_right_keywords(tagger):
    left_files = "cc/*G"
    right_files = "cc/*D"
    left_dict = get_most_used_words(get_words_of_files(left_files, tagger))
    right_dict = get_most_used_words(get_words_of_files(right_files, tagger))
    return left_dict, right_dict
