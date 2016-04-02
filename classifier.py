import re, string
from collections import defaultdict, Counter

def getData():
    with file('train/sanction.cor') as f:
        src = f.read()

#extract each entry in the training data
    l_src = src.split(" \n\n")
    m = list()
    for i in l_src:
        m.append(re.match('\<tag.*?\/\>', i))

#create bag of words for each context. tokenize, lowercase,
#remove stopwords
    l_tok = list()
    l_src = [w.lower() for w in l_src]
#remove empty strings
    l_src = filter(None, l_src)

    return l_src
    
    
def dictTag():
    d_id = dict()

    for i in range(len(getData())):
        v_id = re.search("^\d{6}", getData()[i])
        v_tag = re.search("\"(\d{6})", getData()[i])
        d_id[v_id.group(0)] = v_tag.group(1)

    return d_id
