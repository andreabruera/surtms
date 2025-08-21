import multiprocessing
import matplotlib
import mne
import numpy
import os
import pickle
import random
import scipy
import sklearn

from matplotlib import pyplot
from sklearn import linear_model
from tqdm import tqdm
from utils import read_eegtms_data, read_models_representations, read_models_sims

def process_case(inputs):
    loc = inputs[0]
    elecs = inputs[1]
    mode = inputs[2]
    cond = inputs[3]
    curr_out = os.path.join(out_folder, loc, mode)
    os.makedirs(curr_out, exist_ok=True)

    ### what is common to all approaches is that 
    ### they are time-resolved
    if args.locations == 'searchlight':
        ### preparing time ranges
        time_ranges = [(t*0.1, (t+1)*0.1) for t in range(-4, 20)]
        plot_ts = [_+.05 for _, __ in time_ranges]

    for t_i, t in enumerate(times):


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
            for t in range(len(eeg.times)):
                t_corrs = list()
                for _ in range(50):
                    test_keys = random.sample(keys, k=test_size)
                    train_keys = [k for k in keys if k not in test_keys]
                    #test_eeg = {k : sub_data[k][elecs, t] for k in test_keys}
                    #train_eeg = {k : sub_data[k][elecs, t] for k in train_keys}
                    ### 
                    confound_remover = sklearn.linear_model.LinearRegression()
                    if 'lev' in model:
                        confound_model = 'gpt2-small_surprisal'
                    else:
                        confound_model = 'phonetic_levenshtein'
                    ### removing confound
                    confound_remover.fit(
                              [vecs[confound_model][k] for k in train_keys],
                              [sub_data[k][elecs, t] for k in train_keys],
                              )
                    confound_train = confound_remover.predict(
                              [vecs[confound_model][k] for k in train_keys]
                              )
                    confound_test = confound_remover.predict(
                              [vecs[confound_model][k] for k in test_keys]
                              )
                    train_eeg = {k : v for k, v in zip(train_keys, confound_train)}
                    test_eeg = {k : v for k, v in zip(test_keys, confound_test)}

                    for test_item, test_real in test_eeg.items():
                        rsa_keys = [tuple(sorted([test_item, k])) for k in train_keys if test_item.split('_')[1]!=k.split('_')[1]]
                        pred = numpy.average([model_sims[both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        #pred = numpy.average([model_sims[both_k]*sub_data[k][elecs, t] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        '''
                        denom = numpy.average([model_sims[both_k] for both_k in rsa_keys]) 
                        sub = numpy.average([train_eeg[k]*denom for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        if 'leven' in model:
                            confound = numpy.average([models['gpt2-small_8'][both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                            denom_confound = numpy.average([models['gpt2-small_8'][both_k] for both_k in rsa_keys]) 
                        else:
                            confound = numpy.average([models['phonetic_levenshtein'][both_k]*train_eeg[k] for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                            denom_confound = numpy.average([models['phonetic_levenshtein'][both_k] for both_k in rsa_keys]) 
                        sub_confound = numpy.average([train_eeg[k]*denom_confound for k, both_k in zip(train_keys, rsa_keys)], axis=0)
                        '''
                        corr = scipy.stats.spearmanr(test_eeg[test_item], pred).statistic
                        #corr = scipy.stats.spearmanr(test_real, pred-sub).statistic
                        #corr = scipy.stats.spearmanr(test_real-(confound-sub_confound), pred-sub).statistic
                        t_corrs.append(corr)
                sub_corr.append(numpy.average(t_corrs))
            cond_res[model].append(sub_corr)
    #with open(os.path.join('encoding_plots', loc, mode, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
    with open(os.path.join('..', 'encoding_plots_confound', loc, mode, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
        pickle.dump(cond_res, o)
    fig, ax = pyplot.subplots(constrained_layout=True, figsize=(20, 10))
    ax.hlines(y=0., xmin=min(eeg.times), xmax=max(eeg.times), color='black')
    ax.vlines(x=0., ymin=-.05, ymax=.1, color='black')
    for plot_model, plot_model_data in cond_res.items():
        ax.plot(eeg.times, numpy.average(plot_model_data, axis=0), label=plot_model)
    ax.legend()
    ax.set_xticks([-.2, 0, .2, .4, .6, .8, 1.,])
    ax.set_ylim(bottom=-.05, top=.15)
    ax.set_title(cond)
    pyplot.savefig(os.path.join('..', 'encoding_plots_confound', loc, mode, '{}_{}_encoding.jpg'.format(cond, mode)), dpi=300)
    #pyplot.savefig(os.path.join('encoding_plots', loc, mode, '{}_{}_encoding.jpg'.format(cond, mode)), dpi=300)

parser = argparse.ArgumentParser()
parser.add_argument(
                    '--approach', 
                    required=True,
                    choices=[
                             'rsa', 
                             'rsa_encoding', 
                             'ridge',
                             ]
                    )
parser.add_argument(
                    '--subjects', 
                    required=True,
                    choices=[
                             'within', 
                             'across', 
                             ]
                    )
parser.add_argument(
                    '--residualization', 
                    required=True,
                    choices=[
                             'none', 
                             'cv-confound', 
                             ]
                    )
parser.add_argument(
                    '--locations', 
                    required=True,
                    choices=[
                             'all-only', 
                             'searchlight', 
                             'handmade', 
                             ]
                    )
parser.add_argument(
                    '--debug', 
                    required=False,
                    action='store_true',
                    )
global args
args = parser.parse_args()

### preparing the folder
global out_folder
out_folder = os.path.join(
                          'results',
                          args.approach,
                          '{}_subjects'.format(args.subjects),
                          '{}_resid'.format(args.residualization),
                          args.locations,
                          'debug_{}'.format(args.debug.lower())
                          )
os.makedirs(out_folder, exist_ok=True)

### reading eeg data
eeg_data, times = read_eegtms_data(debug=args.debug)

### reading models
models = read_models_representations()
### reading sims
sims = read_models_sims()

### restricting actual models 
req = [
       #'surprisal', 
       'phonetic_levenshtein', 
       #'frequency', 
       #'fasttext',
       'gpt2-small_8',
       'gpt2-small_surprisal',
       ]
models = {k : models[k] for k in req}
sims = {k : sims[k] for k in req}

### reading clusters
if args.locations == 'all-only':
    locations = read_handmade_clusters(all_only=True)
elif args.locations == 'handmade':
    locations = read_handmade_clusters()
else:
    locations = read_searchlight_clusters()

### preparing multiprocessing arguments
inputs = list()
for loc, elecs in locations.items():
    for mode in tqdm([
                 'noun',
                 'verb',
                 ]):
        for cond, cond_data in tqdm(eeg_data.items()):
            inputs.append((loc, elecs, mode, cond))
import pdb; pdb.set_trace()

### multiprocessing
with multiprocessing.Pool() as p:
    p.map(process_case, inputs)
