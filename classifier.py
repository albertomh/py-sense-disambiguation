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


##################################################################################
### Same code as was used above to fetch training data, here used to get test data.
def getTest():

    with file('test/sanction-p.eval') as f:
        src = f.read()
    l_src = src.split(" \n\n")

    m = list()
    for i in l_src:
        m.append(re.match('\<tag.*?\/\>', i))
    l_tok = list()
    l_src = [w.lower() for w in l_src]
    l_src = filter(None, l_src)

    return l_src

### As was done above with the training data; tokenizing and stripping test data.
def cleanTest():

    d_clean = dict()
    l_pairs = list()
    l_clean = getTest()
    l_stop = ["he", "she", "and", "it", "its", "that", "to", "a", "or", "by", "mine", "my", "your", "yours", "their", "theirs", "theyre", "he", "she", "them", "is", "are", "we", "our", "ours", "his", "her", "i", "if", "but", "be", "will", "the", "like", "has", "have", "mr", "mrs", "with", "not", "in", "of", "for", "was", "when", "they", "all", "an", "were", "had", "also", "as", "no", "what", "tag", "from", "on", "dr", "do", "this", "at", "even"]

    for i in range(len(l_clean)):
        s_id = re.search("(\d{6})", l_clean[i]).group(1)
        l_clean[i] = re.sub("\d|\\n|\&.*?\.|\'s|\<tag\>.*?\<\/\>", "", l_clean[i])
        l_clean[i] = l_clean[i].translate(None, string.punctuation)
        l_clean[i] = re.sub("  | \w{1} ", " ", l_clean[i])
        l_clean[i] = [word for word in l_clean[i].split() if word not in l_stop]
        l_pairs.append((s_id, l_clean[i]))

    return l_pairs
##################################################################################


### Returns probabilities of a word given a sense class for words
### found in the context of the test data.
def lp(i):
    l_p = list()
    for uid in set_val:
        for word in cleanTest()[i][1]:
            if word in cv(1)[uid].keys():
                l_p.append((uid, cv(1)[uid][word]))
    
    return l_p


### accumulate() multiplies together the probabilities of seeing each word featurengiven some sense label. Takes a list as an argument.
### Returns a list of tuples of the form (sense_id, product_of_probabilities)
### ATTRIBUTION: CODE FOR reduce(...): http://bit.ly/158kuNl
def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
       yield key, reduce(operator.mul, (item[1] for item in subiter))


### Multiplies together probabilities of a word feature given a sense class and the probability of the corresponding sense label.
### Fetches the label with the highest probability, stores it as tagid.
### trans looks up tagid in the dict/sanction.dic file by using the ld() function defined below.
###
### Writes a text file of a format identical to the gold standard file for easy comparison later.
### The format follows 'snippet_ID:sense', with each entry on a new line.
def pc(i):
    l_lp = list()
    for (k, v) in lp(i):
        for (k2, v2) in countSID(1).iteritems():
            if k==k2:
                l_lp.append((k, v*v2))

    tagid = max(list(accumulate(l_lp)),key=itemgetter(1))[0]
    trans = [value for key, value in ld().items() if tagid==key]
    s_res = str(cleanTest()[i][0]) + ":" + str(trans[0]) + "\n"

    with open("results.txt", "a") as outfile:
        outfile.write(s_res)


### Looks up tagid in dictionary.
def ld():
    d_tag = dict()
    with file('dict/sanction.dic') as f:
        tagdict = f.read()

        all_tags = re.findall('uid.*?\>', tagdict)

        for i in all_tags:
            uid = re.search('\d{6}', i)
            tag = re.search('tag\=(.*?)\>', i)
            d_tag[uid.group(0)] = tag.group(1)
        return d_tag


### Calculates percentage of correct sense disambiguations and returns percentage value along a list of
### the snippet ID and sense that were correctly classified.
def check():
    gold = [line.rstrip('\n') for line in open('gold/sanction-p')]
    res = [line.rstrip('\n') for line in open('results.txt')]
    for i in range(len(res)):
       res[i] = re.sub('\d{6}\:', "", res[i])

    counter=0
    correct = list()
    for i in range(len(res)):
        if str(res[i]) in gold[i]:
            counter+=1
            correct.append(gold[i])

    return str(round(float(counter)*100/len(res), 3)) + "% of sense disambiguations were correct." + " These corresponded to the following test phrases and meanings:" + str(correct)
