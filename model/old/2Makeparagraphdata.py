import math
import string

#https://stackoverflow.com/questions/4576077/python-split-text-on-sentences
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    sentences = [s.translate(str.maketrans('', '', string.punctuation)) for s in sentences] #Remove punctuation...
    return sentences


import pickle
def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def get_counter(subparagraphs):
  subparagraphs2 = [i.split(" ") for j in subparagraphs for i in j]
  subtot = Counter(i for i in list(itertools.chain.from_iterable(subparagraphs2)))
  return subtot


#----------------------------------------------------------------------------------------------------------


cleanedfilename = 'enwiki-cleaned-2019' #from perl
vocabfilename = 'BOCE.English.400K.vocab'


#----------------------------------------------------------------------------------------------------------

min_words_per_sentence = int(4)
min_sentences_per_paragraph = int(3)


if False: #testing
  file_in = open('enwiki-cleaned', "r") 
  line = file_in.readline()
  line = file_in.readline()
  line = file_in.readline()
  line = file_in.readline()
  sentences = split_into_sentences( line.strip() )
  v = [i for i in sentences if len(i.split(" "))>=min_words_per_sentence]



#load all paragraphs into memory
with open(cleanedfilename) as file_in:
  Fulllist = []
  nlines = 0
  for line in file_in:
    nlines = nlines+1
    sentences = split_into_sentences( line.strip() )
    v = [i for i in sentences if len(i.split(" "))>=min_words_per_sentence]
    #print(v)
    if len(v)>=min_sentences_per_paragraph:
      Fulllist.append(v)


print(nlines)
print(len(Fulllist))


#Segment into training and test
dimtestfile = 300000

trainfilename = 'Paragraphdata-training'
testfilename = 'Paragraphdata-testing'
save_obj(Fulllist[0:-dimtestfile], trainfilename )
save_obj(Fulllist[-dimtestfile:], testfilename )



#Get Vocabulary
n_words = 400000
batchsize = 1e6



import itertools
from collections import Counter


niter = int(len(Fulllist)/batchsize)+1

for i in range(niter):
  if i == 0:
    thebatch = [Fulllist[i] for i in range( int(i * batchsize), int((i+1) * batchsize) )]
    totals = get_counter(thebatch)
  if i > 0:
    thebatch = [Fulllist[i] for i in range( int(i * batchsize), int((i+1) * batchsize) ) if i < len(Fulllist)]
    totals = totals + get_counter(thebatch)



vocab = [item[0] for item in totals.most_common(n_words)]


save_obj(vocab, vocabfilename )



























