import nltk
from nltk.stem.snowball import FrenchStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.parse.stanford import StanfordParser
import nltk.tag.stanford as st
from tree import *
from collections import defaultdict

class FeaturesExtractor:
    """
    This class is able to extract a few features from a french text. It uses the nltk library, together with the stanford french parser and tagger.
    """

    LEFT = [e.split()[-1].lower() for e in [ "François Hollande", "Anne Hidalgo", "Jean-Luc Mélenchon", "Najat Belkacem", "Fleur Pellerin", "Manuel Valls", "Ségolène Royal", "Arnaud Montebourg", "Benoît Hamon", "Laurent Fabius", "Jean-Marc Ayrault", "Pierre Moscovici", "Aurélie Filippetti", "Martine Aubry", "Axelle Lemaire", "Bernard Cazeneuve", "Marisol Touraine", "Jean-Luc Romero", "Stéphane Le Foll", "Claude Bartolone", "Harlem Désir", "Gérard Collomb", "Bertrand Delanoë", "Edgar Morin", "Vincent Peillon", "J-P. Chevènement", "François Rebsamen", "Delphine Batho", "Jean-Chr. Cambadélis", "Bruno Julliard", "Bruno Le Roux", "Caroline De Haas", "Gerard Filoche", "Johanna Rolland", "Elisabeth Guigou", "Marylise Lebranchu", "Jean-Yves Le Drian", "Myriam El Khomri", "Jean-Paul Huchon", "Jean-Jacques Urvoas", "Julien Dray", "Faouzi Lamdaoui", "Michèle Delaunay", "Pierre Laurent", "gaccio bruno", "L Grandguillaume", "Pierre Larrouturou", "Valérie Fourneyron", "laurence rossignol", "Victorin LUREL", "Jean Allemane", "Jean-Claude Amara", "Joëlle Aubron", "Clémentine Autain", "Gérard Bach-Ignasse", "Alain Badiou", "Armand Barbès", "Charlie Bauer", "Helyette Bess", "Marcel Body", "Michel Bounan", "Serge Bricianer", "Claire Brière-Blanchet", "Cornelius Castoriadis", "François Cerutti", "Gabriel Charavay", "Marc Chirik", "Georges Cipriani", "Michel Collinet", "Serge Cosseron", "Julien Coupat", "Yvan Craipeau", "Gilles Dauvé", "Théodore Dézamy", "Alphonse Esquiros", "Didier François", "Alain Geismar", "Clara Geoffroy", "Marina Ginestà", "Pierre Goldman", "Daniel Guérin", "Robert Guiheneuf", "Serge July", "Roger Langlais", "Pierre Lanneret", "Albert Laponneraye", "Guy Lardreau", "Stéphane Lavignotte", "Gérard Lebovici", "Claude Lefort", "Benny Lévy", "Tony Lévy", "Robert Linhart", "Jean-Patrick Manchette", "Gilbert Marquis", "Jean-François Martos", "François Maspero", "Nathalie Ménigon", "Gérard Miller", "René Monzat", "Gilbert Mury", "André Olivier", "Mezioud Ouldamer", "Aris Papathéodorou", "René Riesel", "Olivier Rolin", "Jean-Marc Rouillan", "Barthélemy Sautayra", "Jacques Sauvageot", "Jaime Semprun", "André Senik", "Henri Simon", "Boris Souvarine", "Clara Thalmann-Ensner", "Barack Obama", "Bill Clinton", "Jimmy Carter", "Lyndon Johnson", "John Kennedy", "Harry Truman", "Franklin Roosevelt" ] ]

    RIGHT = [e.split()[-1].lower() for e in [ "Nicolas Sarkozy", "Kosciusko-Morizet", "Alain Juppé", "François Fillon", "Christine Lagarde", "Jean-François Copé", "Bruno Le Maire", "Valérie Pécresse", "Laurent Wauquiez", "Nadine Morano", "Jean-Pierre Raffarin", "Christian Estrosi", "Rachida Dati", "Xavier Bertrand", "christine Boutin", "François Baroin", "N. Dupont-Aignan", "Roselyne Bachelot", "Eric Ciotti", "Benoist Apparu", "Luc CHATEL", "Frédéric Lefebvre", "Michel Barnier", "Eric Woerth", "Roger KAROUTCHI", "Philippe de Villiers", "Guillaume Peltier", "Thierry MARIANI", "Lionel TARDY", "David Lisnard", "Valérie Boyer", "Bruno Retailleau", "Franck Riester", "Michèle Alliot-Marie", "Geoffroy Didier", "Roxane Decorte", "PierreYves Bournazel", "Arnaud Robinet", "Bernard Accoyer", "Charles Beigbeder", "Bernard Debré", "Franck LOUVRIER", "Dominique Bussereau", "Jean-Claude GAUDIN", "Hervé Mariton", "Gérald DARMANIN", "JeanFrédéric Poisson", "Thierry SOLERE", "Jérôme Chartier", "Jean Auguy", "Serge Ayoub", "Serge de Beketch", "Armand Bernardini", "Binet-Valmer", "Émile Bocquillon", "Jean Boissel", "Sixte-Henri de Bourbon-Parme", "Jérôme Bourbon", "Anne Brassié", "Flavien Brenier", "Patrick Buisson", "Pierre Caziot", "Chard", "Frédéric Chatillon", "Jean Chiappe", "Gilbert Collard", "Louis Darquier de Pellepoix", "Gilbert Devèze", "Pierre Dominique", "Guillaume Faye", "Jean Ferré", "Maurice Gaït", "Pierre Gaxotte", "Yves Guérin-Sérac", "Philippe Henriot", "Pierre Hillard", "Jean Jacoby", "Achille Joinard", "Georges Laederich", "Georges Laffly", "Georges Loustaunau-Lacau", "Bernard Lugan", "Jean Mabire", "Jean Madiran", "René Malliavin", "Abel Manouvriez", "Charles Maurras", "Georges de Nantes", "Albert Paraz", "Lucien Pemjean", "Philippe Péninque", "Philippe Pétain", "Georges Pinault", "Léon de Poncins", "Daniel Raffard de Brienne", "Jean Renaud", "René Resciniti de Says", "Hervé Ryssen", "Émile de Saint-Auban", "Alain Sanders", "Alain Soral", "Ralph Soupault", "Pierre de Villemarest", "Donald Trump", "Georges Bush", "Donald Reagan", "Jimmy Carter", "Richard Nixon", "Dwight Eisenhower", ]]

    NEGATION = ["ne", "ni", "n'", "non", "malheureusement", "mauvais", "horrible", "stupide", "catastrophique", "catastrophe", "désastre", "triste", "sinistre", "exécrable", "ignoble", "infect", "abominable", "épouvantable", "détestable", "désagréable", "odieux", "fâcheux", "déplaisant", "pénible", "déplorable", "méchant", "cruel", "minable", "incapable", "inefficace", "dégoûtant", "malheur", "malheureux", "mal", "pas", "minable", "faux", "inexact", "nuisible", "dangeureux", "malsain", "défavorable", "funeste", "pernicieux", "immoral", "corrupteur", "inapproprié", "éprouvante", "troublant", "extrémiste", "extrémistes", "revendique", "bascule", "emmerdes", "emmerdé", "emmerdés", "merde", "nul", "idéologie", "idéologiquement", "anéantir", "anéanti", "anéantis", "danger", "menace", "fasciste", "fascisme", "xénophobe", "xénophobie", "xénophobique", "désarroi"]

    def __init__(self, f, keywords):
        s = f.read()
        self.keywords = keywords
        self.file = s
        self.sentences = sent_tokenize(s)
        self.parser = StanfordParser("stanford-parser-full-2014-08-27/stanford-parser", "stanford-parser-full-2014-08-27/stanford-parser-3.4.1-models")
        self.tagger = st.StanfordPOSTagger("stanford-postagger-full-2014-08-27/models/french.tagger", "stanford-postagger-full-2014-08-27/stanford-postagger.jar")
        self.ner = st.StanfordNERTagger("stanford-ner-2014-08-27/classifiers/english.all.3class.distsim.crf.ser.gz", "stanford-ner-2014-08-27/stanford-ner.jar")

        self.trees = []
        for sent in self.sentences:
            try:
                self.trees.append(self.parser.raw_parse(sent))
            except OSError:
                self.trees.append([])
        self.words = self.word_tokenize_without_punc(s)
        self.stemmer = FrenchStemmer()
        self.stems = [self.stemmer.stem(w) for w in self.words]
        self.words_sentences = [self.word_tokenize_without_punc(s) for s in self.sentences]
        self.tags = self.tagger.tag(self.words)
        self.tags_sentences = [self.tagger.tag([w for w in self.words_sentences[i]]) for i in range(len(self.sentences))]
        self.entities = self.ner.tag(self.words)
        self.entities_sentences = [self.ner.tag([w for w in self.words_sentences[i]]) for i in range(len(self.sentences))]
        self.left_subject = defaultdict(lambda: 0)
        self.left_compl = defaultdict(lambda: 0)
        self.left_neg_subject = defaultdict(lambda: 0)
        self.left_neg_compl = defaultdict(lambda: 0)
        self.right_subject = defaultdict(lambda: 0)
        self.right_compl = defaultdict(lambda: 0)
        self.right_neg_subject = defaultdict(lambda: 0)
        self.right_neg_compl = defaultdict(lambda: 0)
        self.left_ref = 0
        self.right_ref = 0
        self.trees_leaves = []
        for e in self.trees:
            res = []
            extract_leaves(list(e)[0], res)
            self.trees_leaves.append(tuple_to_dict(res))
        self.extract_keywords()

    def extract_keywords(self):
        # get all people
        for key in FeaturesExtractor.LEFT + FeaturesExtractor.RIGHT:
            # for each sentence
            for i in range(len(self.sentences)):
                # if one person is in the sentence
                if key in [word.lower() for word in self.words_sentences[i]]:
                    # for all propositions of the sentence
                    for k in self.trees_leaves[i].keys():
                        # if the person is in the proposition
                        if key in [e[0] for e in self.trees_leaves[i][k]]:
                            # get position of the person in the proposition
                            indexes = [e for e in self.trees_leaves[i][k] if e[0] == key]
                            # get position of the verb in the proposition
                            index_v = [e for e in self.trees_leaves[i][k] if 'V' in e[1]]
                            neg = False
                            for neg_word in FeaturesExtractor.NEGATION:
                                if neg_word == [e[0] for e in self.trees_leaves[i][k]]:
                                    neg = True
                            for index in indexes:
                                if key in FeaturesExtractor.LEFT and index_v:
                                    self.left_ref += 1
                                    if index < index_v[0]:
                                        self.left_subject[key] += 1
                                        if neg:
                                            self.left_neg_subject[key] += 1
                                    else:
                                        self.left_compl[key] += 1
                                        if neg:
                                            self.left_neg_compl[key] += 1
                                if key in FeaturesExtractor.RIGHT and index_v:
                                    self.right_ref += 1
                                    if index < index_v[0]:
                                        self.right_subject[key] += 1
                                        if neg:
                                            self.right_neg_subject[key] += 1
                                    else:
                                        self.right_compl[key] += 1
                                        if neg:
                                            self.right_neg_compl[key] += 1

    def word_tokenize_without_punc(self, s):
        w = word_tokenize(s)
        tags = self.tagger.tag(w)
        return [w[i] for i in range(len(w)) if tags[i][1] != "PUNC"]
        
    def try_function(self, fun, *args):
        try:
            return fun(*args)
        except ZeroDivisionError:
            return 0.

    def extract_features(self):
        res = [
            (self.get_words_in_sentence,),
            (self.get_persons,),
            (self.get_organizations,),
            (self.get_average_negation,),
            (self.get_persons_left_right,),
            (self.get_persons_left,),
            (self.get_exclamation,),
            (self.get_interrogation,),
            (self.get_left_subject,),
            (self.get_right_subject,),
            (self.get_left_compl,),
            (self.get_right_compl,),
            (self.get_left_neg_subject,),
            (self.get_right_neg_subject,),
            (self.get_left_neg_compl,),
            (self.get_right_neg_compl,),
        ]
        for i in range(len(self.keywords)):
            res.append((self.get_keyword, self.keywords[i]))
        return [self.try_function(*e) for e in res]

    def get_left_subject(self):
            return sum([self.left_subject[k] for k in self.left_subject.keys()]) / self.left_ref

    def get_right_subject(self):
            return sum([self.right_subject[k] for k in self.right_subject.keys()]) / self.right_ref

    def get_left_compl(self):
            return sum([self.left_compl[k] for k in self.left_compl.keys()]) / self.left_ref

    def get_right_compl(self):
            return sum([self.right_compl[k] for k in self.right_compl.keys()]) / self.right_ref

    def get_left_neg_subject(self):
            return sum([self.left_neg_subject[k] for k in self.left_neg_subject.keys()]) / self.get_left_subject()

    def get_right_neg_subject(self):
            return sum([self.right_neg_subject[k] for k in self.right_subject.keys()]) / self.get_right_subject()

    def get_left_neg_compl(self):
            return sum([self.left_neg_compl[k] for k in self.left_neg_compl.keys()]) / self.get_left_compl()

    def get_right_neg_compl(self):
            return sum([self.right_neg_compl[k] for k in self.right_neg_compl.keys()]) / self.get_right_compl()

    def get_persons(self):
        """
        Number of persons / number of words
        """
        return len([self.entities[i][0] for i in range(len(self.entities)) if self.entities[i][1] == "PERSON"]) / len(self.words)

    def get_exclamation(self):
        """
        Number of exclamation marks / number of sentences
        """
        return self.file.count("?") / len(self.sentences)

    def get_interrogation(self):
        """
        Number of interrogation marks / number of sentences
        """
        return self.file.count("!") / len(self.sentences)

    def get_persons_left_right(self):
        """
        Number of persons from the left and right list / number of persons
        """
        return len([self.entities[i][0] for i in range(len(self.entities)) if self.entities[i][1] == "PERSON" and (self.entities[i][0].lower() in FeaturesExtractor.LEFT or self.entities[i][0].lower() in FeaturesExtractor.RIGHT)]) / len(self.words)

    def get_persons_left(self):
        """
        Number of persons from the left list / number of persons from the left or right
        """
        return len([self.entities[i][0] for i in range(len(self.entities)) if self.entities[i][1] == "PERSON" and self.entities[i][0].lower() in FeaturesExtractor.LEFT]) / len([self.entities[i][0] for i in range(len(self.entities)) if self.entities[i][1] == "PERSON" and (self.entities[i][0].lower() in FeaturesExtractor.LEFT or self.entities[i][0].lower() in FeaturesExtractor.RIGHT)])

    def get_organizations(self):
        """
        Number of proper nouns / number of words
        """
        return len([self.entities[i][0] for i in range(len(self.entities)) if self.entities[i][1] == "ORGANIZATION"]) / len(self.words)

    def get_keyword(self, keyword):
        """
        Number of keywords / Number of words
        """
        return len(list(filter(lambda x: x == keyword, self.stems))) / len(self.words)

    def get_words_in_sentence(self):
        """
        Average length of sentence / 100
        """
        return len(self.words) / len(self.sentences) / 100

    def get_average_negation(self):
        """
        Number of negated sentences / number of sentences
        """
        return sum([1 if list(filter(lambda x: x in FeaturesExtractor.NEGATION or x[:2] == "n'", s)) else 0 for s in self.words_sentences]) / len(self.sentences)

    def get_negation_keywords_rate(self, keyword):
        """
        Number of negated sentences with keyword / number of sentences with keyword
        """
        k, n = 0, 0
        for i in range(len(self.sentences)):
            if keyword in [self.stemmer.stem(word) for word in self.words_sentences[i]]:
                k += 1
                if list(filter(lambda x: x in FeaturesExtractor.NEGATION, [word.lower() for word in self.words_sentences[i]])):
                    n += 1
        return n / k
