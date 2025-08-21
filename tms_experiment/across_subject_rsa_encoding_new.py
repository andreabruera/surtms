import multiprocessing
import matplotlib
import mne
import numpy
import os
import pickle
import random
import scipy

from matplotlib import pyplot
from tqdm import tqdm

def process_case(inputs):
    loc = inputs[0]
    elecs = inputs[1]
    mode = inputs[2]
    cond = inputs[3]
    #os.makedirs(os.path.join('across_encoding_plots_confound', loc, mode), exist_ok=True)
    os.makedirs(os.path.join('across_encoding_plots', loc, mode), exist_ok=True)
    cond_res = {m : numpy.zeros(shape=(len(eeg.times), 1000)) for m in models.keys()}
    #bootstrap_s = random.choices(list(eeg_data[cond].keys()), k=len(list(eeg_data[cond].keys())))
    bootstrap_s = list(eeg_data[cond].keys())
    #for sub, sub_data in tqdm(eeg_data[cond].items()):
    across_data = dict()
    for sub in bootstrap_s:
        sub_data = eeg_data[cond][sub]
        if len(sub_data.keys()) == 0:
            continue
        for k, v in sub_data.items():
            try:
                across_data[k].append(v)
            except KeyError:
                across_data[k] = [v]
    across_data = {k : numpy.average(v, axis=0) for k, v in across_data.items()}
    for _ in tqdm(range(1000)):
        if mode == 'noun':
            keys = [k for k in across_data.keys() if k.split('_')[0] not in ['Er', 'Sie']]
        if mode == 'verb':
            keys = [k for k in across_data.keys() if k.split('_')[0] in ['Er', 'Sie']]
        ### bootstrap keys
        keys = list(set(random.choices(keys, k=len(keys))))
        test_size = int(len(keys)*0.2)
        for model, model_sims in models.items():
            sub_corr = list()
            for t in range(len(eeg.times)):
                t_corrs = list()
                test_keys = random.sample(keys, k=test_size)
                train_keys = [k for k in keys if k not in test_keys]
                test_eeg = {k : across_data[k][elecs, t] for k in test_keys}
                train_eeg = {k : across_data[k][elecs, t] for k in train_keys}
                for test_item, test_real in test_eeg.items():
                    rsa_keys = [tuple(sorted([test_item, k])) for k in train_keys if test_item.split('_')[1]!=k.split('_')[1]]
                    pred = numpy.average([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                    denom = numpy.average([model_sims[both_k] for both_k in rsa_keys]) 
                    sub = numpy.average([train_eeg[k]*denom for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                    #control = numpy.average([models['phonetic_levenshtein'][both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                    ### normalizing predicted responses
                    corr = scipy.stats.spearmanr(test_real, pred-sub).statistic
                    #corr = scipy.stats.spearmanr(test_real-control, pred-control).statistic
                    cond_res[model][t][_] = corr
    #with open(os.path.join('across_encoding_plots_confound', loc, mode, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
    with open(os.path.join('across_encoding_plots', loc, mode, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
        pickle.dump(cond_res, o)
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
    ax.hlines(y=0., xmin=min(eeg.times), xmax=max(eeg.times), color='black')
    ax.vlines(x=0., ymin=-.05, ymax=.1, color='black')
    for plot_model, plot_model_data in cond_res.items():
        ax.plot(eeg.times, numpy.average(plot_model_data, axis=1), label=plot_model)
    ax.legend()
    #ax.set_xticks([-.4, -.2, 0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
    #ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1.,])
    ax.set_ylim(bottom=-.45, top=.45)
    ax.set_title(cond)
    #pyplot.savefig(os.path.join('across_encoding_plots_confound', loc, mode, '{}_{}_encoding.jpg'.format(cond, mode)), dpi=300)
    pyplot.savefig(os.path.join('across_encoding_plots', loc, mode, '{}_{}_encoding.jpg'.format(cond, mode)), dpi=300)

eeg_data = dict()
folder = os.path.join('surpreegtms', 'derivatives')

with tqdm() as counter:
    for root, direc, fz in os.walk(folder):
        for f in fz:
            if 'sham' not in f:
                continue
            if 'fif' not in f:
                continue
            sub = int(f.split('_')[0].split('-')[-1])
            ses = int(f.split('_')[1].split('-')[-1])
            cond = f.split('_')[3].split('-')[-1]
            if cond not in eeg_data.keys():
                eeg_data[cond] = dict()
            if sub not in eeg_data[cond].keys():
                eeg_data[cond][sub] = dict()
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
            #eeg.crop(tmin=-.2, tmax=2.)
            eeg.crop(tmin=-.2, tmax=1.)
            eeg.decimate(8)
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
                        eeg_data[cond][sub][key].append(curr_data[w_i, :, :])
                    except KeyError:
                        eeg_data[cond][sub][key] = [curr_data[w_i, :, :]]
            counter.update(1)
eeg_data = {cond : {sub : {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in ___.items()} for sub, ___ in __.items()} for cond, __ in eeg_data.items()}

req = [
       'surprisal', 
       'phonetic_levenshtein', 
       #'frequency', 
       'fasttext',
       'gpt2-small_8',
       'gpt2-small_surprisal',
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

locations = {'all' : list()}
counter = dict()
with open('electrode_portions.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        try:
            locations[line[2]].append(int(line[1]))
        except KeyError:
            locations[line[2]] = [int(line[1])]
        locations['all'].append(int(line[1]))

locations = {k : sorted(v) for k, v in locations.items()}
inputs = list()
for loc, elecs in locations.items():
    for mode in tqdm([
                 'noun',
                 'verb',
                 ]):
        for cond, cond_data in tqdm(eeg_data.items()):
            inputs.append((loc, elecs, mode, cond))

with multiprocessing.Pool() as p:
    p.map(process_case, inputs)
