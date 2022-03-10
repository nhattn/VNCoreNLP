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

class Tagger:
    filtered_tags = set(string.punctuation)
    filtered_tags.add(u'\u2026')
    filtered_tags.add(u'\u201d')
    filtered_tags.add(u'\u201c')
    filtered_tags.add(u'\u2019')
    filtered_tags.add('...')
    model = None

    def __init__(self, kernel):
        self.load_kernel(kernel)

    def load_kernel(self, kernel):
        with open(kernel, 'rb') as fin:
            Tagger.model = pickle.load(fin)

    @staticmethod
    def gen_tag(word):
        if has_tagged(word):
            word_tag = word.split('/')
            tag = word_tag.pop()
            word = '/'.join(word_tag)
            return [(word, tag)]
        return [(word, 'N')]

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
                sentence.extend(Tagger.gen_tag(token))
            train_data.append(sentence)
        if len(train_data) == 0:
            return
        sentences = [ Tagger.sent2features(sent, True) for sent in train_data ]
        labels = [ Tagger.sent2labels(sent) for sent in train_data ]
        X_train, X_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.01, random_state=42)
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
        f1_scorer = make_scorer(metrics.flat_f1_score,average='weighted', labels=TAGS)
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
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'word[:1].isdigit()': word[:1].isdigit(),
            'word[:3].isupper()': word[:3].isupper(),
            'word.isfiltered': word in Tagger.filtered_tags,
        }
        if i > 0:
            word1 = sent[i - 1][0] if is_training else sent[i - 1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word[:1].isdigit()': word1[:1].isdigit(),
                '-1:word[:3].isupper()': word1[:3].isupper(),
            })
            if i > 1:
                word2 = sent[i - 2][0] if is_training else sent[i - 2]
                features.update({
                    '-2:word.lower()': word2.lower(),
                    '-2:word.istitle()': word2.istitle(),
                    '-2:word.isupper()': word2.isupper(),
                })
        else:
            features['BOS'] = True
        if i < len(sent) - 1:
            word1 = sent[i + 1][0] if is_training else sent[i + 1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word[:1].isdigit()': word1[:1].isdigit(),
                '+1:word.isupper()': word1.isupper(),
            })
            if i < len(sent) - 2:
                word2 = sent[i + 2][0] if is_training else sent[i + 2]
                features.update({
                    '+2:word.lower()': word2.lower(),
                    '+2:word.istitle()': word2.istitle(),
                    '+2:word.isupper()': word2.isupper(),
                })
        else:
            features['EOS'] = True

        return features

    @staticmethod
    def sent2features(sent, is_training):
        return [Tagger.word2features(sent, i, is_training) for i in range(len(sent))]

    @staticmethod
    def postagging(str):
        return Tagger.postagging_tokens(str.split(' '))

    @staticmethod
    def postagging_tokens(tokens):
        labels = Tagger.model.predict([Tagger.sent2features(tokens, False)])
        return tokens, labels[0]
