import multiprocessing
import pickle
import matplotlib
import numpy
import os
import pickle
import random
import scipy

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


def mp_read(mc):

    data = dict()
    data[mode] = {cond : dict()}

    ### reading data
    with tqdm() as counter:
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
    return data

font_setup('../../fonts')

mapper = {
       'surprisal' : 'word surprisal (5-gram)', 
       #'surprisal_lemma', 
       'phonetic_levenshtein' : 'phonological similarity', 
       #'phonetic_length', 
       #'frequency', 
       #'fasttext',
       #'gpt2-small_0',
       #'gpt2-small_4',
       'gpt2-small_8' : 'semantic similarity (GPT2)',
       #'xlm-large_20',
       'gpt2-small_surprisal' : 'word surprisal (GPT2)',
       #'gpt2-small_entropy',
       #'gpt2-large_0',
       #'gpt2-large_14',
       #'gpt2-large_32',
       #'fasttext_lemma',
       }

req = [
       'surprisal', 
       #'surprisal_lemma', 
       'phonetic_levenshtein', 
       #'phonetic_length', 
       #'frequency', 
       #'fasttext',
       #'gpt2-small_0',
       #'gpt2-small_4',
       'gpt2-small_8',
       #'xlm-large_20',
       'gpt2-small_surprisal',
       #'gpt2-small_entropy',
       #'gpt2-large_0',
       #'gpt2-large_14',
       #'gpt2-large_32',
       #'fasttext_lemma',
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

'''
mp_args = list()

for mode in os.listdir('eeg_sims'):
    #data[mode] = dict()
    for cond in os.listdir(os.path.join('eeg_sims', mode)):
        #data[mode][cond] = dict()
        if cond != 'p':
            mp_args.append((mode, cond))
with multiprocessing.Pool() as p:
    res = p.map(mp_read, mp_args)


data = dict()
for ind_data in res:
    for k, v in ind_data.items():
        if k not in data.keys():
            data[k] = v
        else:
            for k_two, v_two in v.items():
                data[k][k_two] = v_two
'''
data = dict()
count = 0
### reading data
with tqdm() as counter:
    for mode in os.listdir('notms_eeg_sims'):
        data[mode] = dict()
        for cond in os.listdir(os.path.join('notms_eeg_sims', mode)):
            data[mode][cond] = dict()
            for f in os.listdir(os.path.join('notms_eeg_sims', mode, cond)):
                #if count >= 1:
                #    continue
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
                count += 1
with open(os.path.join('plots', 'notms_latest_pickles.pkl'), 'wb') as o:
    pickle.dump(data, o)
'''
with open(os.path.join('plots', 'latest_pickles.pkl'), 'rb') as o:
    data = pickle.load(o)
'''
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
        os.makedirs(os.path.join('notms_plots', mode, elecs), exist_ok=True)
        for cond, cond_data in tqdm(elecs_data.items()):
            cond_data = {k : v for k, v in cond_data.items() if len(v)>1}
            if len(cond_data.keys()) == 0:
                continue
            for k, v in cond_data.items():
                assert len(v) > 1
                lengths.add(len(v))
            avg_eeg = {k : numpy.average(v, axis=0) for k, v in cond_data.items()}
            results[mode][elecs][cond] = dict()
            keys = sorted(cond_data.keys())
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
                print(len(current_keys))
                results[mode][elecs][cond][model] = list()
                for t in tqdm(range(len(times))):
                    try:
                        eeg_sims = [avg_eeg[k][t] for k in current_keys]
                    except IndexError:
                        results[mode][elecs][cond][model].append(-0.05)
                    model_sims = [model_data[k] for k in current_keys]
                    corr = scipy.stats.spearmanr(eeg_sims, model_sims, nan_policy='omit').statistic
                    results[mode][elecs][cond][model].append(corr)
            #cond_res = numpy.nanmean(numpy.array(cond_res), axis=1)
            fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
            ax.hlines(y=0., xmin=min(times), xmax=max(times), color='black')
            ax.vlines(x=0., ymin=-.2, ymax=.2, color='black')
            ax.set_ylim(top=0.3, bottom=-.1)
            #results[mode][elecs][cond] = dict()
            #for model, model_res in cond_res.items():
            for model, model_res in results[mode][elecs][cond].items():
                actual_model = mapper[model]
                try:
                    ax.plot(times, model_res, label=actual_model)
                except ValueError:
                    continue
            ax.legend(fontsize=23)
            ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1., 1.2, 1.4, 1.6, 1.8, 2.])
            pyplot.xticks(fontsize=18)
            pyplot.yticks(fontsize=18)
            pyplot.ylabel('Spearman rho', fontsize=21, fontweight='bold')
            pyplot.xlabel('Seconds after stimulus', fontsize=21, fontweight='bold')
            ax.set_title(
                    'Condition:   {}\nPosition:   {}'.format(cond, mode), 
                         fontsize=35, 
                         fontweight='bold',
                         )
            pyplot.savefig(os.path.join('notms_plots', mode, elecs, '{}_{}.jpg'.format(cond, mode)), dpi=300)
            #pyplot.savefig(os.path.join('plots', mode, '{}_{}.jpg'.format(cond, mode)), dpi=300)
print(lengths)
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

for mode, mode_res in results.items():
    for elecs, elecs_res in mode_res.items():
        for cond, cond_res in elecs_res.items():
'''
