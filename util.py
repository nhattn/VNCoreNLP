# coding: utf-8

TAGS = ['Nc','Ny','Np','Nu','A','C','E','I','L','M','N','P','R','S','T','V','X','F']

PUNCTUATION = "@`#$%&~|[]<>'(){}*+-=;,?.!:\"/"

def has_tagged(word):
    for tag in TAGS:
        tag = '/'+tag
        if word[-len(tag):] == tag:
            return True
    return False

def normalize_text(text):
    if '/B_W' in text or '/I_W' in text:
        tokens = text.strip().split(' ')
        text = ''
        for tok in tokens:
            word_tag = tok.split('/')
            tag = word_tag.pop()
            word = '/'.join(word_tag)
            if tag == 'I_W':
                text += '_'
            else:
                text += ' '
            text += word
    for t in TAGS:
        text = text.replace('/'+t+' ',' ')
    return text.strip()

# https://stackoverflow.com/a/2077321
def unicode_replace(text):
    uni = [
        ["…","..."],
        ["“","\""],
        ["”","\""],
        ["–","-"],
        [""," "]
    ]
    for _, c in enumerate(uni):
        text = text.replace(c[0],c[1])
    return text

def regx(text):
    ans = unicode_replace(text)
    for ch in PUNCTUATION:
        ans = ans.replace(ch,' ' + ch + ' ')
    return ans

def is_word(word):
    if word in PUNCTUATION:
        return False
    if word.isdigit():
        return False
    for ch in PUNCTUATION + '1234567890':
        if ch in word:
            return False
    return True

def tokenize(text):
    ans = regx(text)
    ans += " "
    ans = ans.replace("_", " ")
    return [ w.strip() for w in ans.split(' ') if w.strip() ]
