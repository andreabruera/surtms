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
        'fasttext_lemma' : dict(),
        #'fasttext_phrase' : dict(),
        'surprisal' : dict(), 
        'surprisal_lemma' : dict(), 
        'frequency' : dict(),
        'frequency_lemma' : dict(),
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
    for __, w_two in ws:
        #if w_one == w_two:
        #    #print([w_one, w_two])
        #    continue
        if not check_ws(w_one, w_two):
            continue
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        try:
            sims['phonetic_length'][key]
            continue
        except KeyError:
            pass
        ipa_one = epi.transliterate(w_one)
        ipa_two = epi.transliterate(w_two)
        sim = -abs(len(ipa_one)-len(ipa_two))
        #key = tuple(sorted([w_one, w_two]))
        sims['phonetic_length'][key] = sim
        ### similarity!
        sim = -levenshtein(ipa_one, ipa_two)
        sims['phonetic_levenshtein'][key] = sim
        #sim = 1 - scipy.spatial.distance.cosine(ft['{} {}'.format(_, w_one)], ft['{} {}'.format(__, w_two)])
        #key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #sims['fasttext_phrase'][key] = sim

### cn
with open('/data/u_bruera_software/word_vectors/de/conceptnet_de.pkl', 'rb') as i:
    cn = pickle.load(i)
for _, w_one in lemma_ws:
    old_one = w_one
    for __, w_two in lemma_ws:
        if not check_ws(w_one, w_two):
            continue
        sim = 1 - scipy.spatial.distance.cosine(cn[old_one.lower()], cn[w_two.lower()])
        if w_one in mapper.keys():
            w_one = mapper[w_one]
        if w_two in mapper.keys():
            w_two = mapper[w_two]
        if _ in mapper.keys():
            _ = mapper[_]
        if __ in mapper.keys():
            __ = mapper[__]
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #key = tuple(sorted([w_one, w_two]))
        sims['conceptnet'][key] = sim
        #sim = 1 - scipy.spatial.distance.cosine(ft['{} {}'.format(_, w_one)], ft['{} {}'.format(__, w_two)])
        #key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #sims['fasttext_phrase'][key] = sim

### ft
ft = fasttext.load_model(os.path.join('/', 'data', 'u_bruera_software', 'word_vectors','de', 'cc.de.300.bin'))
for _, w_one in ws:
    for __, w_two in ws:
        if not check_ws(w_one, w_two):
            continue
        #if w_one == w_two:
        #    #print([w_one, w_two])
        #    continue
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        ### similarity!
        sim = 1 - scipy.spatial.distance.cosine(ft[w_one], ft[w_two])
        #key = tuple(sorted([w_one, w_two]))
        sims['fasttext'][key] = sim
        #sim = 1 - scipy.spatial.distance.cosine(ft['{} {}'.format(_, w_one)], ft['{} {}'.format(__, w_two)])
        #key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #sims['fasttext_phrase'][key] = sim
        
for _, w_one in lemma_ws:
    for __, w_two in lemma_ws:
        if not check_ws(w_one, w_two):
            continue
        #if w_one == w_two:
        #    #print([w_one, w_two])
        #    continue
        ### similarity!
        sim = 1 - scipy.spatial.distance.cosine(ft[w_one], ft[w_two])
        if w_one in mapper.keys():
            w_one = mapper[w_one]
        if w_two in mapper.keys():
            w_two = mapper[w_two]
        if _ in mapper.keys():
            _ = mapper[_]
        if __ in mapper.keys():
            __ = mapper[__]
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #key = tuple(sorted([w_one, w_two]))
        sims['fasttext_lemma'][key] = sim
        #sim = 1 - scipy.spatial.distance.cosine(ft['{} {}'.format(_, w_one)], ft['{} {}'.format(__, w_two)])
        #key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        #sims['fasttext_phrase'][key] = sim

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
    for __, w_two in ws:
        if not check_ws(w_one, w_two):
            continue
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        try:
            sims['frequency'][key]
            continue
        except KeyError:
            pass
        try:
            one = freqs[w_one]
        except KeyError:
            one = 0
        try:
            two = freqs[w_two]
        except KeyError:
            two = 0
        #if w_one == w_two:
        #    continue
        sim = -abs(one-two)
        #key = tuple(sorted([w_one, w_two]))
        sims['frequency'][key] = sim
for _, w_one in lemma_ws:
    for __, w_two in lemma_ws:
        if not check_ws(w_one, w_two):
            continue
        try:
            one = freqs[w_one]
        except KeyError:
            one = 0
        try:
            two = freqs[w_two]
        except KeyError:
            two = 0
        #if w_one == w_two:
        #    continue]
        sim = -abs(one-two)
        if w_one in mapper.keys():
            w_one = mapper[w_one]
        if w_two in mapper.keys():
            w_two = mapper[w_two]
        if _ in mapper.keys():
            _ = mapper[_]
        if __ in mapper.keys():
            __ = mapper[__]
        #key = tuple(sorted([w_one, w_two]))
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        sims['frequency_lemma'][key] = sim
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
    try:
        cooc_one = coocs[vocab[_]][vocab[w_one]]
    except KeyError:
        cooc_one = 0
    for __, w_two in ws:
        try:
            cooc_two = coocs[vocab[__]][vocab[w_two]]
        except KeyError:
            cooc_two = 0
        if not check_ws(w_one, w_two):
            continue
        ### similarity!
        sim = -abs(cooc_one-cooc_two)
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        sims['surprisal'][key] = sim
for _, w_one in lemma_ws:
    try:
        cooc_one = coocs[vocab[_]][vocab[w_one]]
    except KeyError:
        cooc_one = 0
    for __, w_two in lemma_ws:
        try:
            cooc_two = coocs[vocab[__]][vocab[w_two]]
        except KeyError:
            cooc_two = 0
        if not check_ws(w_one, w_two):
            continue
        sim = -abs(cooc_one-cooc_two)
        if w_one in mapper.keys():
            w_one = mapper[w_one]
        if w_two in mapper.keys():
            w_two = mapper[w_two]
        if _ in mapper.keys():
            _ = mapper[_]
        if __ in mapper.keys():
            __ = mapper[__]
        key = tuple(sorted(['{}_{}'.format(_, w_one), '{}_{}'.format(__, w_two)]))
        sims['surprisal_lemma'][key] = sim

out_f = 'models'
os.makedirs(out_f, exist_ok=True)
for model, res in sims.items():
    with open(os.path.join(out_f, '{}.tsv'.format(model)), 'w') as o:
        for ks, sim in res.items():
            o.write('{}\t{}\t{}\n'.format(ks[0], ks[1], sim))
