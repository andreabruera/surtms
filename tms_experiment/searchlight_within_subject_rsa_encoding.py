import argparse
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

def align_events(eeg_events, tsv_events):

    if len(eeg_events) == len(tsv_events):
        new_tsv_events = [i for i in range(len(tsv_events))]
        new_eeg_events = [i for i in range(len(eeg_events))]
    else:
        if len(eeg_events) < len(tsv_events):
            new_tsv_events = list()
            new_eeg_events = [i for i in range(len(eeg_events))]
            counter = 0
            for w_i in range(len(eeg_events)):
                tsv_trg = int(tsv_events[w_i+counter][header.index('trigger')])
                eeg_trg = eeg_events[w_i][-1]
                if tsv_trg == eeg_trg:
                    new_tsv_events.append(w_i+counter)
                else:
                    ### we don´t know how many events we need to skip...
                    for _ in range(1, 10):
                        counter += 1
                        tsv_trg = int(tsv_events[w_i+counter][header.index('trigger')])
                        eeg_trg = eeg_events[w_i][-1]
                        if tsv_trg == eeg_trg:
                            new_tsv_events.append(w_i+counter)
                            break
                        else:
                            pass

        elif len(eeg_events) > len(tsv_events):
            new_eeg_events = list()
            new_tsv_events = [i for i in range(len(tsv_events))]
            counter = 0
            for w_i in range(len(tsv_events)):
                tsv_trg = int(tsv_events[w_i][header.index('trigger')])
                eeg_trg = eeg_events[w_i+counter][-1]
                if tsv_trg == eeg_trg:
                    new_eeg_events.append(w_i+counter)
                else:
                    counter += 1
                    new_eeg_events.append(w_i+counter)

    return new_eeg_events, new_tsv_events

def process_case(inputs):
    loc = inputs[0]
    elecs = inputs[1]
    mode = inputs[2]
    cond = inputs[3]
    folder = 'searchlight_encoding_plots'
    if args.confound:
        folder = '{}_confound'.format(folder)
    out_folder = os.path.join(folder, loc, mode)
    os.makedirs(out_folder, exist_ok=True)

    ### preparing time ranges
    time_ranges = [(t*0.1, (t+1)*0.1) for t in range(-4, 20)]
    plot_ts = [_+.05 for _, __ in time_ranges]

    cond_res = dict()
    for model, model_sims in tqdm(models.items()):
        #z_model_sims = scipy.stats.zscore([model_sims[k] for k in sorted(model_sims.keys())])
        #model_sims = {k : v for k, v in zip(sorted(model_sims.keys()), z_model_sims)}
        cond_res[model] = list()
        for sub, sub_data in tqdm(eeg_data[cond].items()):
            if len(sub_data.keys()) == 0:
                continue
            if mode == 'noun':
                keys = [k for k in sub_data.keys() if k.split('_')[0] not in ['Er', 'Sie']]
            if mode == 'verb':
                keys = [k for k in sub_data.keys() if k.split('_')[0] in ['Er', 'Sie']]
            #keys = [k for k in keys if k in model_sims.keys()]
            test_size = int(len(keys)*0.2)
            sub_corr = list()
            #for t in range(len(eeg.times)):
            for start, end in time_ranges:
                t_idxs = [t_i for t_i, t in enumerate(eeg.times) if t>=start and t<=end]
                t_corrs = list()
                for _ in range(50):
                    test_keys = random.sample(keys, k=test_size)
                    train_keys = [k for k in keys if k not in test_keys]
                    test_eeg = {k : sub_data[k][elecs, :][:, t_idxs].flatten() for k in test_keys}
                    train_eeg = {k : sub_data[k][elecs, :][:, t_idxs].flatten() for k in train_keys}
                    for test_item, test_real in test_eeg.items():
                        rsa_keys = [tuple(sorted([test_item, k])) for k in train_keys if test_item.split('_')[1]!=k.split('_')[1]]
                        pred = numpy.average([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        denom = numpy.average([model_sims[both_k] for both_k in rsa_keys]) 
                        sub = numpy.average([train_eeg[k]*denom for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        if args.confound:
                            if 'leven' in model:
                                confound = numpy.average([models['gpt2-small_8'][both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                                denom_confound = numpy.average([models['gpt2-small_8'][both_k] for both_k in rsa_keys]) 
                            else:
                                confound = numpy.average([models['phonetic_levenshtein'][both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                                denom_confound = numpy.average([models['phonetic_levenshtein'][both_k] for both_k in rsa_keys]) 
                            sub_confound = numpy.average([train_eeg[k]*denom_confound for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                            corr = scipy.stats.spearmanr(test_real-(confound-sub_confound), pred-sub).statistic
                        else:
                            corr = scipy.stats.spearmanr(test_real, pred-sub).statistic
                        t_corrs.append(corr)
                sub_corr.append(numpy.average(t_corrs))
            cond_res[model].append(sub_corr)
    with open(os.path.join(out_folder, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
        pickle.dump(cond_res, o)
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
    ax.hlines(y=0., xmin=min(plot_ts), xmax=max(plot_ts), color='black')
    ax.vlines(x=0., ymin=-.05, ymax=.1, color='black')
    for plot_model, plot_model_data in cond_res.items():
        ax.plot(plot_ts, numpy.average(plot_model_data, axis=0), label=plot_model)
    ax.legend()
    #ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1.,])
    ax.set_xticks(plot_ts)
    ax.set_ylim(bottom=-.05, top=.15)
    ax.set_title(cond)
    pyplot.savefig(
                   os.path.join(
                                out_folder, 
                                '{}_{}_encoding.jpg'.format(
                                                            cond, 
                                                            mode,
                                                            )
                                ), 
                                dpi=300,
                                )

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--confound', 
                    action='store_true', 
                    required=False,
                    )
global args
args = parser.parse_args()

locations = read_searchlight_clusters()

eeg_data = dict()
folder = os.path.join('surpreegtms', 'derivatives')

with tqdm() as counter:
    for root, direc, fz in os.walk(folder):
        for f in fz:
            #if 'sham' not in f:
            #    continue
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
            eeg = mne.read_epochs(
                                  os.path.join(root, f), 
                                  preload=True, 
                                  verbose=False,
                                  )
            #eeg.crop(tmax=1.)
            #eeg.decimate(8)
            curr_data = eeg.get_data()
            ### aligning events
            eeg_events_idxs, tsv_events_idxs = align_events(eeg.events, tsv_events)
            assert len(eeg_events_idxs) == len(tsv_events_idxs)
            for t_i, e_i in zip(tsv_events_idxs, eeg_events_idxs):
                tsv_trg = int(tsv_events[t_i][header.index('trigger')])
                eeg_trg = eeg.events[e_i][-1]
                assert tsv_trg == eeg_trg
                if tsv_events[t_i][header.index('trigger')][-1] in ['2', '6']:
                    continue
                if 'NA' in tsv_events[t_i]:
                    continue
                curr_w = tsv_events[t_i][header.index('word')]
                if tsv_events[t_i][header.index('trigger')][-1] == '5':
                    diff = 1
                else:
                    diff = 2
                prev_w = tsv_events[t_i-diff][header.index('word')]
                key = '{}_{}'.format(prev_w, curr_w,)
                try:
                    eeg_data[cond][sub][key].append(curr_data[e_i, :, :])
                except KeyError:
                    eeg_data[cond][sub][key] = [curr_data[e_i, :, :]]
            counter.update(1)
eeg_data = {cond : {sub : {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in ___.items()} for sub, ___ in __.items()} for cond, __ in eeg_data.items()}

req = [
       #'surprisal', 
       'phonetic_levenshtein', 
       #'frequency', 
       #'fasttext',
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
