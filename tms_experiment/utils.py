import mne
import numpy
import os
import scipy

from tqdm import tqdm

def align_events(eeg_events, tsv_events, header):

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


def read_eegtms_data(debug=False):
    eeg_data = dict()
    folder = os.path.join('..', 'surpreegtms', 'derivatives')

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
                if debug:
                    eeg.crop(tmax=1.)
                    eeg.decimate(8)
                curr_data = eeg.get_data()
                ### aligning events
                eeg_events_idxs, tsv_events_idxs = align_events(eeg.events, tsv_events, header)
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
                    key = '{}_{}'.format(prev_w, curr_w)
                    try:
                        eeg_data[cond][sub][key].append(curr_data[e_i, :, :])
                    except KeyError:
                        eeg_data[cond][sub][key] = [curr_data[e_i, :, :]]
                counter.update(1)
    ### averaging across the two sessions
    eeg_data = {cond : 
                   {sub : 
                       {k : numpy.average(v, axis=0) if len(v)>1 else v[0] for k, v in ___.items()} 
                            for sub, ___ in __.items()} 
                                 for cond, __ in eeg_data.items()}
    return eeg_data, eeg.times

def read_models_sims():
    folder = os.path.join(
                         '..', 
                         'general_resources', 
                         'models_sims',
                         )
    ### reading sims
    sims = dict()
    for f in os.listdir(folder):
        m = f.split('.')[0]
        sims[m] = dict()
        with open(os.path.join(folder, f)) as i:
            for l in i:
                line = l.strip().split('\t')
                sims[m][(line[0], line[1])] = float(line[2])
    return sims

def read_models_representations():
    folder = os.path.join(
                         '..', 
                         'general_resources', 
                         'models_representations',
                         )
    vecs = dict()
    for f in os.listdir(folder):
        m = f.split('.')[0]
        vecs[m] = dict()
        with open(os.path.join(folder, f)) as i:
            for l in i:
                line = l.strip().split('\t')
                vecs[m][line[0]] = numpy.array(line[1:], dtype=numpy.float64)
                assert len(vecs[m][line[0]].shape) == 1 
    return vecs

def read_handmade_clusters(all_only=False):
    f = os.path.join(
                         '..', 
                         'general_resources', 
                         'electrode_portions.tsv',
                         )
    locations = {'all' : list()}
    counter = dict()
    with open(f) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                continue
            if not all_only:
                try:
                    locations[line[2]].append(int(line[1]))
                except KeyError:
                    locations[line[2]] = [int(line[1])]
            locations['all'].append(int(line[1]))

    locations = {k : sorted(v) for k, v in locations.items()}
    return locations

def read_searchlight_clusters(cluster_size=5):
    idxs = dict()
    with open(
              os.path.join(
                           '..',
                           'general_resources',
                           'electrode_portions.tsv',
                           )) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                continue
            idxs[line[0]] = int(line[1])

    locations = dict()
    with open(
              os.path.join(
                  'resources',
                  'avg_electrode_distances.tsv',
                  )) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            cluster_elecs = [idxs[v.split(',')[0]] for v in line[1:cluster_size]]
            locations[line[0]] = cluster_elecs+[idxs[line[0]]]

    for k, v in locations.items():
        assert len(v) == 5
    return locations
