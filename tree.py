import sys
from nltk.parse.stanford import StanfordParser
from collections import defaultdict

def extract_leaves(tree, res=[], depth=0, label=None):
    if isinstance(tree, str):
        res.append((depth, tree.lower(), label))
        return
    for e in tree:
        extract_leaves(e, res, depth + 1, tree.label())

def tuple_to_dict(t):
    res = defaultdict(lambda: [])
    for e in t:
        res[e[0]].append((e[1], e[2]))
    return res

if __name__ == "__main__":
    parser = StanfordParser("stanford-parser-full-2014-08-27/stanford-parser", "stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models")
    trees = list(parser.raw_parse("Il faut que Sarkozy mange Fillon."))
    res = []
    extract_leaves(trees[0], res)
    print(res)
    print (tuple_to_dict(res))
