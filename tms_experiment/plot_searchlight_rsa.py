import argparse
import matplotlib
import mne
import numpy
import os
import pickle
import scipy

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager, pyplot
from tqdm import tqdm

def font_setup(font_folder):
    ### Font setup
    # Using Helvetica as a font
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

parser = argparse.ArgumentParser()
parser.add_argument('--confound', action='store_true', required=False)
args = parser.parse_args()


font_setup('../../fonts')

#time_ranges = [(t*0.1, (t+1)*0.1) for t in range(-4, 20)]
#times = [_+.05 for _, __ in time_ranges]
times = [_*.1 for _ in range(11)]

idxs = dict()
with open('electrode_portions.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        idxs[line[0]] = int(line[1])
pos = dict()
for sub in [1, 2]:
    for ses in [1, 2]:
        montage_file = os.path.join('electrode_positions', 'MorphSem2_vp{:02}_s{}.csv'.format(sub, ses))
        with open(montage_file) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split(',')
                if 'VUNTER' in line:
                    elec = 'EOGV'
                elif 'VOBEN' in line:
                    elec = 'EOGH'
                elif 'HL' in line or 'HR' in line:
                    continue
                else:
                    elec = line[1].replace('Z', 'z').replace('FP', 'Fp')
                if elec not in pos.keys():
                    pos[elec] = list()
                pos[elec].append(numpy.array(
                                        line[2:], 
                                        dtype=numpy.float32,
                                            ))
pos = {k : numpy.average(v, axis=0) for k, v in pos.items()}
for k, v in pos.items():
    #print(v)
    assert v.shape == (3,)
montage = mne.channels.make_dig_montage(ch_pos=pos)
count = 0
for root, direc, fz in os.walk('surpreegtms/derivatives'):
    for f in fz:
        if 'fif' not in f:
            continue
        if count >= 1:
            continue
        eeg = mne.read_epochs(os.path.join(root, f), preload=True, verbose=False)
        times = eeg.times
        count += 1
eeg.set_montage(montage)

idxs = dict()
with open('electrode_portions.tsv') as i:
    for l_i, l in enumerate(i):
        line = l.strip().split('\t')
        if l_i == 0:
            continue
        idxs[line[0]] = int(line[1])

locations = dict()
with open('electrode_distances.tsv') as i:
    for l_i, l in enumerate(i):
        if l_i == 0:
            continue
        line = l.strip().split('\t')
        cluster_elecs = [idxs[v.split(',')[0]] for v in line[1:5]]
        locations[line[0]] = cluster_elecs+[idxs[line[0]]]
#adj = numpy.full(shape=(len(locations.keys()), len(locations.keys())), fill_value=False)
rows = list()
cols = list()
vals = list()
for idx, ones in locations.items():
    for one in ones[:-1]:
        rows.append(ones[-1])
        cols.append(one)
        vals.append(1)
    #adj[ones[-1]][ones[:-1]] = True
    #adj[ones[:-1]][ones[-1]] = True
adj = scipy.sparse.coo_matrix((vals, (rows, cols)), shape=(61, 61))

res = dict()
folder = 'searchlight_encoding_plots'
if args.confound:
    folder = '{}_confound'.format(folder)

with tqdm() as counter:
    for root, direc, fz in os.walk(folder):
        for f in fz:
            if '.pkl' not in f:
                continue
            with open(os.path.join(root, f), 'rb') as i:
                curr_res = pickle.load(i)
            #curr_res = {k : curr_res[k] for k in ['phonetic_levenshtein', 'surprisal']}
            cond = f.split('_')[-3]
            if cond not in res.keys():
                res[cond] = dict()
            mode = f.split('_')[-2]
            if mode not in res[cond].keys():
                res[cond][mode] = dict()
            loc = root.split('/')[-2]
            for model, model_res in curr_res.items():
                if model not in res[cond][mode].keys():
                    res[cond][mode][model] = numpy.zeros(shape=(61, 24, 10))
                model_res = numpy.array(model_res)
                assert model_res.shape in [(24, 24), (24, 10)]
                res[cond][mode][model][idxs[loc]] = model_res[:, 4:14]
                #res[cond][mode][model][idxs[loc]] = model_res
for cond, cond_res in res.items():
    for mode, mode_res in cond_res.items():
        for model, model_res in mode_res.items():
            out_folder = os.path.join(folder.replace('encoding',  'scalp'), cond, mode)
            os.makedirs(out_folder, exist_ok=True)
            out_f = os.path.join(out_folder, '{}.jpg'.format(model))
            model_res = numpy.swapaxes(model_res, 0, 1)
            p_res = numpy.swapaxes(model_res, 1, 2)
            assert p_res.shape == (24, 10, 61)
            evo = mne.EvokedArray(
                                  numpy.average(model_res, axis=0), 
                                  mne.create_info(ch_names=eeg.ch_names, sfreq=10, ch_types='eeg').set_montage(montage), 
                                  tmin=0.) 
            #import pdb; pdb.set_trace()
            if 'lev' in model:
                c_ = 'gray'
                c__ = 'black'
            elif '8' in model:
                c_ = 'orange'
                c__ = 'darkorange'
            else:
                c_ = 'mediumpurple'
                c__ = 'rebeccapurple'
            cmap = LinearSegmentedColormap.from_list(
                                           "mycmap", 
                                          [
                                           'white',
                                           'white',
                                           c_,
                                           c__,
                                           ])
            ts, _, ps, __ = mne.stats.spatio_temporal_cluster_1samp_test(
                                  p_res,
                                  tail=1,
                                  threshold=dict(start=0, step=0.2), 
                                  n_jobs=os.cpu_count()-1, 
                                  adjacency=adj,
                                  max_step=1,
                                  n_permutations=10000,
                                  )
            ps = ps.reshape(ts.shape).T
            evo.plot_topomap(
                    #ch_type='eeg', 
                    #time_unit='s', 
                    times=[0.,.1,.2,.3,.4,.5,.6,.7,.8, .9],
                    nrows=1, 
                    ncols=10,
                    scalings={'eeg':1.}, 
                    cmap=cmap,
                    vlim=(0., 0.1),
                    contours=2,
                    #sphere='eeglab',
                    #sensors=False,
                    mask=ps<=0.05,
                    outlines='head',
                    sphere=(0, 10, 0, 90),
                    mask_params=dict(
                                     marker='*', 
                                     markerfacecolor='black', 
                                     markeredgecolor='black',
                                     linewidth=0, 
                                     markersize=10),
                    #size = 3.,
                    )

            print(out_f)
            pyplot.savefig(out_f, dpi=600)
            pyplot.savefig(out_f.replace('jpg', 'svg'))
            pyplot.clf()
            pyplot.close()
