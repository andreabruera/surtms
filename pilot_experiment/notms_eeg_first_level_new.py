import multiprocessing
import matplotlib
import mne
import numpy
import os
import scipy

from matplotlib import pyplot
from tqdm import tqdm

eeg_data = dict()
folder = os.path.join('surpreeg', 'derivatives')

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
            eeg.crop(tmin=-.2, tmax=2.)
            curr_data = eeg.get_data()
            if len(eeg.events) == len(tsv_events):
                for w_i, w in enumerate(tsv_events):
                    if w[header.index('trigger')][-1] in ['2', '6']:
                        continue
                    if 'NA' in w:
                        continue
                    curr_w = w[header.index('word')]
                    #if w[header.index('trigger')][-1] == '5':
                    if w[header.index('trigger')][-1] == '4':
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

'''
out = 'erps'
os.makedirs(out, exist_ok=True)
idxs = [eeg.ch_names.index(c) for c in ['C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2', 'FC1', 'FCz',  'FC2']]
for cond, cond_data in eeg_data.items():
    for sub, sub_data in cond_data.items():
        for k, v in sub_data.items():
            if 'Er_' not in k and 'Sie_' not in k:
                continue
            counter = 0
            for e in v:
                counter += 1
                fig, ax = pyplot.subplots()
                ax.plot(eeg.times, numpy.average(e[idxs, :], axis=0))
                pyplot.savefig(os.path.join(out, '{}_{}_{}_{}.jpg'.format(k, sub, cond, counter)))
                pyplot.close()
'''

eeg_data = {cond : {sub : {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in ___.items()} for sub, ___ in __.items()} for cond, __ in eeg_data.items()}

global locations
locations = {'all' : list()}
counter = dict()
with open('notms_electrode_portions.tsv') as i:
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

def multi(sub):
    first_levels = dict()
    for loc, elec_idxs in locations.items():
        first_levels[loc] = dict()
        for cond, cond_data in tqdm(eeg_data.items()):
            first_levels[loc][cond] = list()
            #for sub, sub_data in tqdm(cond_data.items()):
            try:
                sub_data = cond_data[sub]
            except KeyError:
                continue
            print(sub_data.keys())
            ts = dict()
            #for t_i in tqdm(range(2)):
            for t_i in tqdm(range(len(eeg.times))):
                for k_one_i, k_one in enumerate(sorted(sub_data.keys())): 
                    for k_two_i, k_two in enumerate(sorted(sub_data.keys())): 
                        if k_two_i <= k_one_i:
                            continue
                        try:
                            assert k_one.split('_')[0] not in ['Er', 'Sie'] and k_one.split('_')[0] not in ['Er', 'Sie']
                        except AssertionError:
                            try:
                                assert k_one.split('_')[0] in ['Er', 'Sie'] and k_one.split('_')[0] in ['Er', 'Sie']
                            except AssertionError:
                                continue

                        one = sub_data[k_one][elec_idxs, t_i]
                        two = sub_data[k_two][elec_idxs, t_i]
                        sim = scipy.stats.spearmanr(one, two).statistic
                        try:
                            ts[tuple(sorted([k_one, k_two]))].append(sim)
                        except KeyError:
                            ts[tuple(sorted([k_one, k_two]))] = [sim]
            first_levels[loc][cond] = ts
    return first_levels

with multiprocessing.Pool() as m:
    #res = m.map(multi, range(1, 29))
    res = m.map(multi, range(1, 13))

for sub, sub_res in enumerate(res):
    for loc, loc_data in sub_res.items():
        for cond, cond_data in loc_data.items():
            if type(cond_data) != dict:
                print([sub, cond])
                continue
            #out_fold = os.path.join('eeg_sims', str(loc), str(cond))
            out_fold = os.path.join('notms_eeg_sims', str(loc), str(cond))
            os.makedirs(out_fold, exist_ok=True)
            if len(cond_data.keys()) == 0:
                continue
            with open(os.path.join(out_fold, 'sub-{:02}.tsv'.format(sub+1)), 'w') as o:
                o.write('word_one\tword_two\t')
                for t in eeg.times:
                    o.write('{}\t'.format(t))
                o.write('\n')
                for ws, ts in cond_data.items():
                    o.write('{}\t{}\t'.format(ws[0], ws[1]))
                    for t in ts:
                        o.write('{}\t'.format(t))
                    o.write('\n')
