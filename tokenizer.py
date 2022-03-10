# coding: utf-8

import os
import re
import unicodedata as ud
import pickle, string
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
import scipy.stats
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

class Tokenizer:
    bi_grams = set()
    tri_grams = set()
    model = None
    try:
        with open(os.path.join(os.path.dirname(__file__), 'models', 'vocab.txt'), 'r', encoding='utf-8') as fin:
            for token in fin.read().split('\n'):
                tmp = token.split(' ')
                if len(tmp) == 2:
                    bi_grams.add(token)
                elif len(tmp) == 3:
                    tri_grams.add(token)
    except:
        pass
    def __init__(self, kernel):
        self.load_kernel(kernel)

    def load_kernel(self, kernel):
        with open(kernel, 'rb') as fin:
            Tokenizer.model = pickle.load(fin)

    @staticmethod
    def gen_tag(word):
        syllables = word.split('_')
        if not any(syllables):
            return [(word, 'B_W')]
        output = [(syllables[0], 'B_W')]
        for item in syllables[1:]:
            output.append((item, 'I_W'))
        return output

    @staticmethod
    def sent2labels(sent):
        return [label for _, label in sent]

    @staticmethod
    def train(data, saveTo):
        train_data = []
        for row in data:
            tokens = row.split(' ')
            sentence = []
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                sentence.extend(Tokenizer.gen_tag(token))
            train_data.append(sentence)
        if len(train_data) == 0:
            return
        sentences = [ Tokenizer.sent2features(sent, True) for sent in train_data ]
        labels = [ Tokenizer.sent2labels(sent) for sent in train_data ]
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0, random_state=42)
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
        }
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            max_iterations=100,
            all_possible_transitions=True,
            c1=params_space['c1'],
            c2=params_space['c2']
        )
        f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted', labels=['B_W', 'I_W'])
        rs = RandomizedSearchCV( crf, params_space,cv=5,verbose=1, n_jobs=12, n_iter=50, scoring=f1_scorer)
        rs.fit(X_train, y_train)
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        with open(saveTo, 'wb') as pkl:
            pickle.dump(rs.best_estimator_, pkl)

    @staticmethod
    def word2features(sent, i, is_training):
        word = sent[i][0] if is_training else sent[i]
        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
        }
        if i > 0:
            word1 = sent[i - 1][0] if is_training else sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:word.bi_gram()': ' '.join([word1, word]).lower() in Tokenizer.bi_grams,
            })
            if i > 1:
                word2 = sent[i - 2][0] if is_training else sent[i - 2]
                features.update({
                    '-2:word.tri_gram()': ' '.join([word2, word1, word]).lower() in Tokenizer.tri_grams,
                })
        if i < len(sent) - 1:
            word1 = sent[i + 1][0] if is_training else sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:word.bi_gram()': ' '.join([word, word1]).lower() in Tokenizer.bi_grams,
            })
            if i < len(sent) - 2:
                word2 = sent[i + 2][0] if is_training else sent[i + 2]
                features.update({
                    '+2:word.tri_gram()': ' '.join([word, word1, word2]).lower() in Tokenizer.tri_grams,
                })
        return features

    @staticmethod
    def sent2features(sent, is_training):
        return [Tokenizer.word2features(sent, i, is_training) for i in range(len(sent))]

    @staticmethod
    def sylabelize(text):
        text = ud.normalize('NFC', text)

        specials = ["==>", "->", "\.\.\.", ">>",'\n']
        digit = "\d+([\.,_]\d+)+"
        email = "([a-zA-Z0-9_.+-]+@([a-zA-Z0-9-]+\.)+[a-zA-Z0-9-]+)"
        web = "\w+://[^\s]+"
        word = "\w+"
        non_word = "[^\w\s]"
        abbreviations = [
            "[A-ZĐ]+\.",
            "Tp\.",
            "Mr\.", "Mrs\.", "Ms\.",
            "Dr\.", "ThS\."
        ]

        patterns = []
        patterns.extend(abbreviations)
        patterns.extend(specials)
        patterns.extend([web, email])
        patterns.extend([digit, non_word, word])

        patterns = "(" + "|".join(patterns) + ")"
        if isinstance(patterns,bytes):
            patterns = patterns.decode('utf-8')
        tokens = re.findall(patterns, text, re.UNICODE)

        return text, [token[0] for token in tokens]

    @staticmethod
    def tokenize(str):
        text, tmp = Tokenizer.sylabelize(str)
        if len(tmp) == 0:
            return str
        labels = Tokenizer.model.predict([Tokenizer.sent2features(tmp, False)])
        output = tmp[0]
        for i in range(1, len(labels[0])):
            if labels[0][i] == 'I_W' and tmp[i] not in string.punctuation and\
                            tmp[i-1] not in string.punctuation and\
                    not tmp[i][0].isdigit() and not tmp[i-1][0].isdigit()\
                    and not (tmp[i][0].istitle() and not tmp[i-1][0].istitle()):
                output = output + '_' + tmp[i]
            else:
                output = output + ' ' + tmp[i]
        return output
