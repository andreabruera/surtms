import pickle
import matplotlib
import numpy
import os
import random
import scipy

from matplotlib import pyplot
from tqdm import tqdm

req = [
       'surprisal', 
       #'surprisal_lemma', 
       'phonetic_levenshtein', 
       'frequency', 
       #'frequency_lemma', 
       'fasttext',
       #'fasttext_lemma',
       'conceptnet',
       ]

### reading sims
models = dict()
for f in os.listdir('models'):
    m = f.split('.')[0]
    if m not in req:
        continue
    models[m] = dict()
    with open(os.path.join('models', f)) as i:
        for l in i:
            line = l.strip().split('\t')
            models[m][(line[0], line[1])] = float(line[2])

v_durations = list()
n_durations = list()
for f in os.listdir('durations'):
    with open(os.path.join('durations', f)) as i:
        for l in i:
            line = l.strip().split(';')
            v_durations.append(numpy.sum(numpy.array(line[2:4], dtype=numpy.float32)))
            n_durations.append(float(line[4]))
x_noun = numpy.average(v_durations)
x_task = numpy.average(n_durations)

data = dict()

### reading data
with tqdm() as counter:
    for mode in os.listdir('notms_eeg_sims'):
        data[mode] = dict()
        for cond in os.listdir(os.path.join('notms_eeg_sims', mode)):
            data[mode][cond] = dict()
            for f in os.listdir(os.path.join('notms_eeg_sims', mode, cond)):
                with open(os.path.join('notms_eeg_sims', mode, cond, f)) as i:
                    for l_i, l in enumerate(i):
                        line = l.strip().split('\t')
                        if l_i == 0:
                            times = numpy.array(line[2:], dtype=numpy.float64)
                            continue
                        try:
                            data[mode][cond][(line[0], line[1])].append(numpy.array(line[2:], dtype=numpy.float64))
                        except KeyError:
                            data[mode][cond][(line[0], line[1])] = [numpy.array(line[2:], dtype=numpy.float64)]
                counter.update(1)
lengths = set()
results = dict()
modes = [
         'noun', 
         'verb', 
         #'both',
         ]

for mode in tqdm(modes):
    results[mode] = dict()
    for elecs, elecs_data in tqdm(data.items()):
        results[mode][elecs] = dict()
        for cond, cond_data in tqdm(elecs_data.items()):
            for k, v in cond_data.items():
                assert len(v) > 1
                lengths.add(len(v))
            avg_eeg = {k : numpy.average(v, axis=0) for k, v in cond_data.items()}
            results[mode][elecs][cond] = dict()
            keys = sorted([k for k in cond_data.keys() if 'NA' not in k])
            #keys = sorted(cond_data.keys())
            for model, model_data in tqdm(models.items()):
                current_keys = [k for k in keys if k[0].split('_')[1]!=k[1].split('_')[1]]
                #current_keys = keys.copy()
                #if model != 'surprisal':
                #    current_keys = [k for k in keys if k[0].split('_')[1]!=k[1].split('_')[1]]
                #else:
                current_keys = [k for k in current_keys if tuple(sorted(k)) in model_data.keys()]
                if mode == 'noun':
                    current_keys = [k for k in current_keys if k[0].split('_')[0] not in ['Er', 'Sie'] and k[1].split('_')[0] not in ['Er', 'Sie']]
                elif mode == 'verb':
                    current_keys = [k for k in current_keys if k[0].split('_')[0] in ['Er', 'Sie'] and k[1].split('_')[0] in ['Er', 'Sie']]
                #print(len(current_keys))
                results[mode][elecs][cond][model] = list()
                for t in tqdm(range(len(times))):
                    eeg_sims = [avg_eeg[k][t] for k in current_keys]
                    model_sims = [model_data[k] for k in current_keys]
                    corr = scipy.stats.spearmanr(eeg_sims, model_sims, nan_policy='omit').statistic
                    results[mode][elecs][cond][model].append(corr)
                    '''
                    iters = list()
                    #for _ in range(1000):
                    for _ in range(1000):
                        reduced = random.choices(current_keys, k=len(current_keys))
                        eeg_sims = [avg_eeg[k][t] for k in reduced]
                        model_sims = [model_data[k] for k in reduced]
                        corr = scipy.stats.spearmanr(eeg_sims, model_sims).statistic
                        iters.append(corr)
                    results[mode][cond][model].append(iters)

for mode, mode_res in results.items():
    os.makedirs(os.path.join('plots', mode), exist_ok=True)
    for cond, cond_res in mode_res.items():
        cond_res = numpy.array(cond_res)
        with open(os.path.join('plots', mode, '{}_{}_bootstrap.pkl'.format(cond, mode)), 'wb') as o:
            pickle.dump(cond_res, o)
'''

for mode, mode_res in results.items():
    for elecs, elecs_res in mode_res.items():
        os.makedirs(os.path.join('notms_plots', mode, elecs), exist_ok=True)
        for cond, cond_res in elecs_res.items():
            #cond_res = numpy.nanmean(numpy.array(cond_res), axis=1)
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            ax.hlines(y=0., xmin=min(times), xmax=max(times), color='black')
            ax.vlines(x=0., ymin=-.2, ymax=.2, color='black')
            if mode == 'verb':
                ax.vlines(x=x_noun, ymin=-.1, ymax=.1, color='black')
            else:
                ax.vlines(x=x_task, ymin=-.1, ymax=.1, color='black')
            ax.set_ylim(top=0.3, bottom=-.1)
            for model, model_res in cond_res.items():
                ax.plot(times, model_res, label=model)
            ax.legend()
            ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
            ax.set_title(cond)
            pyplot.savefig(os.path.join('notms_plots', mode, elecs, '{}_{}.jpg'.format(cond, mode)), dpi=300)
            #pyplot.savefig(os.path.join('plots', mode, '{}_{}.jpg'.format(cond, mode)), dpi=300)
print(lengths)
