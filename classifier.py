import re, string, itertools, operator
from collections import defaultdict, Counter
from operator import itemgetter
from itertools import groupby

def getData():
    with file('train/sanction.cor') as f:
        src = f.read()
### Extract each entry in the training data.
    l_src = src.split(" \n\n")

    m = list()
    for i in l_src:
        m.append(re.match('\<tag.*?\/\>', i))
### Create bag of words for each context. Tokenize and lowercase.
    l_tok = list()
    l_src = [w.lower() for w in l_src]
    l_src = filter(None, l_src)

    return l_src


### Returns a dictionary of the form test_id:tag_id
### Used later on in countSID() to count instances of each sense ID
def dictTag():
    d_id = dict()
    for i in range(len(getData())):
        v_id = re.search("^\d{6}", getData()[i])
        v_tag = re.search("\"(\d{6})", getData()[i])
        d_id[v_id.group(0)] = v_tag.group(1)

    return d_id


### Returns a list of tuples of the form (sense_id, news_item) where the news_item
### has been tokenized and stripped of punctuation and stopwords.
def cleanData():
    d_clean = dict()
    l_pairs = list()
    l_clean = getData()
    l_stop = ["he", "she", "and", "it", "its", "that", "to", "a", "or", "by", "mine", "my", "your", "yours", "their", "theirs", "theyre", "he", "she", "them", "is", "are", "we", "our", "ours", "his", "her", "i", "if", "but", "be", "will", "the", "like", "has", "have", "mr", "mrs", "with", "not", "in", "of", "for", "was", "when", "they", "all", "an", "were", "had", "also", "as", "no", "what", "tag", "from", "on", "dr", "do", "this", "at", "even"]

    for i in range(len(l_clean)):
        s_id = re.search("\"(\d{6})", l_clean[i]).group(1)
        l_clean[i] = re.sub("\d|\\n|\&.*?\.|\'s", "", l_clean[i])
        l_clean[i] = l_clean[i].translate(None, string.punctuation)
        l_clean[i] = re.sub("  | tag | \w{1} ", " ", l_clean[i])
        l_clean[i] = [word for word in l_clean[i].split() if word not in l_stop]
        l_pairs.append((s_id, l_clean[i]))
        
    return l_pairs


### Variable storing list of all possible sense IDs (tags).
set_val = set(dictTag().values())


### Returns a dictionary of the form {'sense_id': probability} where probability
### is the prior probability of each sense class, smoothed according to parameter L.
def countSID(L):
    d_cSID = dict()
    set_val = set(dictTag().values())
    for i in set_val:
            d_cSID[i] = float(dictTag().values().count(i)+L)/float(len(cleanData())+L*len(set_val))
            
    return d_cSID


### cv(L) calculates the probability of each word given a sense class.
### Returns d2, where d2 is a dictionary of the form {sense_id1: {word: probability, w: p}, sense_id2: {w: p, w: p}. . .}
### Smoothing is set by the parameter L.
def cv(L):
    global set_val
    res = defaultdict(list)
    for k, v in cleanData():
        res[k].append(v)

    res = dict(res)
    for i in set_val:
        res[str(i)] = [item for sublist in res[str(i)] for item in sublist]

    d1 = dict()
    d2 = dict()
    for i in set_val:
        d1[i] = (dict((word, float(res[i].count(word))) for word in set(res[i])))
        d2[i] = (dict((word, float(res[i].count(word)+L)/float(sum(d1[str(i)].values())+L*len(d1[i]))) for word in set(res[i])))

    return d2
