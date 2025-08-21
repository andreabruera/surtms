import multiprocessing
import matplotlib
import mne
import numpy
import os
import random
import scipy

from matplotlib import pyplot
from tqdm import tqdm

def z_score(data, test_items):
    avg_data = numpy.average([v for k, v in data.items() if k not in test_items], axis=0)
    std_data = numpy.std([v for k, v in data.items() if k not in test_items], axis=0)
    current_data = {k : (v-avg_data)/std_data for k, v in data.items()}
    return current_data

eeg_data = dict()
folder = os.path.join('surpreegtms', 'derivatives')

with tqdm() as counter:
    for root, direc, fz in os.walk(folder):
        for f in fz:
            if 'fif' not in f:
                continue
            sub = int(f.split('_')[0].split('-')[-1])
            ses = int(f.split('_')[1].split('-')[-1])
            cond = f.split('_')[3].split('-')[-1]
            if cond not in eeg_data.keys():
                eeg_data[cond] = dict()
            ev_f = f.replace('eeg-epo.fif', 'events.tsv')
            tsv_events = list()
            triggers = set()
            with open(os.path.join(root, ev_f)) as i:
                for l_i, l in enumerate(i):
                    line = l.strip().split('\t')
                    if l_i == 0:
                        header = line.copy()
                        continue
                    assert len(line) == len(header)
                    triggers.add(int(line[header.index('trigger')]))
                    tsv_events.append(line)
            eeg = mne.read_epochs(os.path.join(root, f), preload=True, verbose=False)
            eeg.decimate(10)
            curr_data = eeg.get_data()
            if len(eeg.events) == len(tsv_events):
                for w_i, w in enumerate(tsv_events):
                    if w[header.index('trigger')][-1] in ['2', '6']:
                        continue
                    if 'NA' in w:
                        continue
                    curr_w = w[header.index('word')]
                    if w[header.index('trigger')][-1] == '5':
                        diff = 1
                    else:
                        diff = 2
                    prev_w = tsv_events[w_i-diff][header.index('word')]
                    key = '{}_{}'.format(prev_w, curr_w)
                    try:
                        eeg_data[cond][key].append(curr_data[w_i, :, :])
                    except KeyError:
                        eeg_data[cond][key] = [curr_data[w_i, :, :]]
            counter.update(1)
eeg_data = {cond : {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in __.items()} for cond, __ in eeg_data.items()}

req = [
       'surprisal', 
       'phonetic_levenshtein', 
       #'frequency', 
       #'fasttext',
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

for mode in tqdm([
             'noun',
             'verb',
             ]):
    os.makedirs(os.path.join('encoding_plots', mode), exist_ok=True)
    for cond, sub_data in tqdm(eeg_data.items()):
        cond_res = dict()
        for model, model_sims in tqdm(models.items()):
            #z_model_sims = scipy.stats.zscore([model_sims[k] for k in sorted(model_sims.keys())])
            #model_sims = {k : v for k, v in zip(sorted(model_sims.keys()), z_model_sims)}
            cond_res[model] = list()
            if mode == 'noun':
                keys = [k for k in sub_data.keys() if k.split('_')[0] not in ['Er', 'Sie']]
            if mode == 'verb':
                keys = [k for k in sub_data.keys() if k.split('_')[0] in ['Er', 'Sie']]
            #keys = [k for k in keys if k in model_sims.keys()]
            test_size = int(len(keys)*0.2)
            sub_corr = list()
            for t in range(len(eeg.times)):
                t_corrs = list()
                for _ in range(100):
                    test_keys = random.sample(keys, k=test_size)
                    train_keys = [k for k in keys if k not in test_keys]
                    test_eeg = {k : sub_data[k][:, t] for k in test_keys}
                    train_eeg = {k : sub_data[k][:, t] for k in train_keys}
                    #train_eeg = z_score(train_eeg, test_keys)
                    train_eeg = z_score(train_eeg, [])
                    for test_item, test_real in test_eeg.items():
                        rsa_keys = [tuple(sorted([test_item, k])) for k in train_keys if test_item.split('_')[1]!=k.split('_')[1]]
                        #num = numpy.sum([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        denom = numpy.average([abs(model_sims[both_k]) for both_k in rsa_keys]) 
                        #pred = num / denom
                        pred = numpy.average([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        sub = numpy.average([train_eeg[k]*denom for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        corr = scipy.stats.spearmanr(test_real, pred-sub)
                        #rsa_keys = [tuple(sorted([test_item, k])) for k in train_keys]
                        #num = numpy.sum([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        #denom = sum([abs(model_sims[both_k]) for both_k in rsa_keys]) 
                        #pred = num / denom
                        #pred = numpy.average([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        #corr = scipy.stats.spearmanr(test_real, pred)
                        t_corrs.append(corr)
                sub_corr.append(numpy.average(t_corrs))
            cond_res[model].append(sub_corr)
        fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
        ax.hlines(y=0., xmin=min(eeg.times), xmax=max(eeg.times), color='black')
        ax.vlines(x=0., ymin=-.2, ymax=.2, color='black')
        for plot_model, plot_model_data in cond_res.items():
            ax.plot(eeg.times, numpy.average(plot_model_data, axis=0), label=plot_model)
        ax.legend()
        ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1., 1.2])
        ax.set_title(cond)
        pyplot.savefig(os.path.join('encoding_plots', mode, '{}_{}_encoding_across.jpg'.format(cond, mode)), dpi=300)
