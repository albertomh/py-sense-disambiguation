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
