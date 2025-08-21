import epitran
import fasttext
import numpy
import os
import pickle
import scipy
import spacy

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])

sp = spacy.load("de_core_news_lg")

ws = list()
lemma_ws = list()
mapper = dict()
with open('transcriptions.txt') as i:
    for l in i:
        line = l.strip().split(';')
        ws.append((line[1], line[2]))
        ws.append((line[2], line[4]))
        lemma = [w.lemma_ for w in sp(' '.join(line[1:]))][1]
        if lemma[-1] != 'n':
            lemma = manual[lemma]
        if lemma == 'issen':
            lemma = 'essen'
        if lemma == 'befähren':
            lemma = 'befahren'
        if lemma == 'zerbrichen':
            lemma = 'zerbrechen'
        if lemma == 'bestichen':
            lemma = 'bestechen'
        if lemma == 'hänselen':
            lemma = 'hänseln'
        if lemma == 'vergissen':
            lemma = 'vergessen'
        if lemma == 'wäschn':
            lemma = 'wäschen'
        if lemma == 'grüßen':
            lemma = 'grüssen'
        if lemma == 'verlässen':
            lemma = 'verlassen'
        mapper[lemma] = line[2]
        lemma_ws.append((line[1], lemma))
        lemma_ws.append((lemma, line[4]))
        manual = {
                  'siebt' : 'sieben',
                  'toastet' : 'toasten',
                  'schient' : 'scheinen',
                  'vergisst' : 'vergessen',
                  'jätet' : 'jäten',
                  }

sims = {
        'fasttext' : dict(),
        'surprisal' : dict(), 
        'frequency' : dict(),
        'phonetic_length' : dict(),
        'phonetic_levenshtein' : dict(),
        'conceptnet' : dict()
        }

epi = epitran.Epitran('deu-Latn')

def check_ws(w_one, w_two):
    check = True
    if 'NA' in w_one:
        check = False
    if 'NA' in w_two:
        check = False
    #if w_two[0].islower()==True and w_one[0].islower()==False:
    #    check = False
    #if w_two[0].islower()==False and w_one[0].islower()==True:
    #    check = False
    if w_one == w_two:
        check = False
    return check


### levenshtein
for _, w_one in ws:
    ipa_one = epi.transliterate(w_one)
    sims['phonetic_length']['{}_{}'.format(_, w_one)] = [len(ipa_one)]
    for __, w_two in ws:
        if not check_ws(w_one, w_two):
            continue
        ipa_two = epi.transliterate(w_two)
        ### similarity!
        sim = -levenshtein(ipa_one, ipa_two)
        try:
            sims['phonetic_levenshtein']['{}_{}'.format(_, w_one)].append(sim)
        except KeyError:
            sims['phonetic_levenshtein']['{}_{}'.format(_, w_one)] = [sim]
for k in sims['phonetic_levenshtein'].keys():
    sims['phonetic_levenshtein'][k] = [numpy.average(sims['phonetic_levenshtein'][k])]

### cn
with open('/data/u_bruera_software/word_vectors/de/conceptnet_de.pkl', 'rb') as i:
    cn = pickle.load(i)
for _, w_one in lemma_ws:
    if 'NA' in [_, w_one]:
        continue
    sims['conceptnet']['{}_{}'.format(_, w_one)] = cn[w_one.lower()]

### ft
ft = fasttext.load_model(os.path.join('/', 'data', 'u_bruera_software', 'word_vectors','de', 'cc.de.300.bin'))
for _, w_one in ws:
    if 'NA' in [_, w_one]:
        continue
    sims['fasttext']['{}_{}'.format(_, w_one)] = ft[w_one]
        
### coocs and freqs
case = 'cased'
lang = 'de'
f = 'wac'
min_count = 10

base_folder = os.path.join('..', '..', 'counts',
                       lang, 
                       f,
                       )
with open(os.path.join(
                        base_folder,
                       '{}_{}_{}_word_freqs.pkl'.format(
                                                         lang, 
                                                         f,
                                                         case
                                                         ),
                       ), 'rb') as i:
    freqs = pickle.load(i)
### freq
for _, w_one in ws:
    if 'NA' in [_, w_one]:
        continue
    try:
        one = freqs[w_one]
    except KeyError:
        one = 0
    sims['frequency']['{}_{}'.format(_, w_one)] = [one]
vocab_file = os.path.join(
                        base_folder,
                       '{}_{}_{}_vocab_min_{}_all-pos.pkl'.format(
                                                           lang, 
                                                           f,
                                                           case,
                                                           min_count
                                                           ),
                       )
with open(vocab_file, 'rb') as i:
    vocab = pickle.load(i)
print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
print('total size of the vocabulary: {:,} words'.format(max(vocab.values())))
coocs_file = os.path.join(base_folder,
              #'{}_{}_forward-coocs_{}_min_{}_win_20.pkl'.format(
              '{}_{}_forward-coocs_{}_min_{}_win_5_all-pos.pkl'.format(
                                                                 lang,
                                                                 f,
                                                                 case,
                                                                 min_count
                                                                 )
                  )
with open(coocs_file, 'rb') as i:
    coocs = pickle.load(i)
### coocs
for _, w_one in ws:
    if 'NA' in [_, w_one]:
        continue
    try:
        cooc_one = coocs[vocab[_]][vocab[w_one]]
    except KeyError:
        cooc_one = 0
    sims['surprisal']['{}_{}'.format(_, w_one)] = [sim]

out_f = 'vecs'
os.makedirs(out_f, exist_ok=True)
for model, res in sims.items():
    with open(os.path.join(out_f, '{}.tsv'.format(model)), 'w') as o:
        for ks, vec in res.items():
            o.write('{}\t'.format(ks))
            for dim in vec:
                o.write('{}\t'.format(dim))
            o.write('\n')
