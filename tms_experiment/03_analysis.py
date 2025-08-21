import argparse
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
from utils import read_eegtms_data, read_handmade_clusters, read_models_representations, read_models_sims

def process_case(inputs):
    loc = inputs[0]
    elecs = inputs[1]
    mode = inputs[2]
    cond = inputs[3]
    curr_out = os.path.join(out_folder, loc, mode)
    os.makedirs(curr_out, exist_ok=True)

    cond_res = dict()
    ### what is common to all approaches is that 

    # they are time-resolved
    if args.locations == 'searchlight':
        t_min = round(min(times*10))
        t_max = round(max(times*10))
        time_ranges = [(
                        (t*0.1)+.05, 
                        t*0.1, 
                        (t+1)*0.1
                        ) for t in range(min_t, max_t)]
    else:
        time_ranges = [(t, t, t) for t in times]
    if args.approach == 'rsa':
        iterations = 1000
        #if args.subjects == 'across':
        #    iterations = 1000
        #else:
        #    iterations = 1
    else:
        iterations = 50
    ### creating output container
    n_ts = len(time_ranges)
    n_subs = len(eeg_data[cond].keys())
    n_models = len(models.keys())
    sorted_models = sorted(models.keys())
    res = numpy.zeros(shape=(n_models, n_subs, n_ts, iterations))
    ### doing pairwise similarities
    if args.approach == 'rsa':
        rsa_keys = set()
        sub_keys = dict()
        eeg_sims = dict()
        with tqdm() as counter:
            for sub, sub_data in tqdm(eeg_data[cond].items()):
                eeg_sims[sub] = dict()
                sub_keys[sub] = set()
                assert len(sub_data.keys()) > 0
                if mode == 'noun':
                    keys = [k for k in sorted(sub_data.keys()) if k.split('_')[0] not in ['Er', 'Sie']]
                if mode == 'verb':
                    keys = [k for k in sorted(sub_data.keys()) if k.split('_')[0] in ['Er', 'Sie']]
                for t_i, ts in enumerate(time_ranges):
                    t = ts[0]
                    t_min = ts[1]
                    t_max = ts[2]
                    t_idxs = [t_i for t_i, _ in enumerate(times) if _>=t_min and _<=t_max]
                    for k_i, k_one in enumerate(keys):
                        for k_two_i, k_two in enumerate(keys):
                            if k_two_i <= k_i:
                                continue
                            key = tuple(sorted((k_one, k_two)))
                            rsa_keys.add(k_one)
                            rsa_keys.add(k_two)
                            sub_keys[sub].add(k_one)
                            sub_keys[sub].add(k_two)
                            corr = scipy.stats.spearmanr(
                                               sub_data[k_one][elecs, :][:, t_idxs],
                                               sub_data[k_two][elecs, :][:, t_idxs],
                                               ).statistic
                            try:
                                eeg_sims[sub][key].append(corr)
                            except KeyError:
                                eeg_sims[sub][key] = [corr]
                            counter.update(1)

    for iteration in tqdm(range(iterations)):
        if args.approach == 'rsa':
            iter_subs = random.choices(
                             list(eeg_data[cond].keys()), 
                             k=len(eeg_data[cond].keys()),
                             )
        else:
            iter_subs = list(eeg_data[cond].keys())
        ### when across subjects, we average
        if args.subjects == 'across':
            sub_data = dict()
            if args.approach == 'rsa':
                all_sub_data = eeg_sims
            else:
                all_sub_data = eeg_data[cond]
            for sub in iter_subs:
                for k, v in all_sub_data[sub].items():
                    try:
                        sub_data[k].append(v)
                    except KeyError:
                        sub_data[k] = [v]
            sub_data = {k : numpy.average(v, axis=0) for k, v in sub_data.items()}
            iter_subs = [0]
            
        for t_i, ts in enumerate(time_ranges):
            t = ts[0]
            t_min = ts[1]
            t_max = ts[2]
            t_idxs = [t_i for t_i, _ in enumerate(times) if _>=t_min and _<=t_max]
            for sub_i, sub in enumerate(iter_subs):
                if args.subjects == 'across':
                    pass
                else:
                    if args.approach == 'rsa':
                        sub_data = eeg_sims[sub]
                    else:
                        sub_data = eeg_data[cond][sub]
                assert len(sub_data.keys()) > 0
                ### subsampling trials
                if args.approach == 'rsa':
                    if args.subjects == 'across':
                        keys = sorted(set(random.choices(list(rsa_keys), k=len(rsa_keys))))
                    else:
                        keys = sorted(set(random.choices(list(sub_keys[sub]), k=len(sub_keys[sub]))))
                        #keys = list(sub_keys[sub])

                else:
                    if mode == 'noun':
                        keys = [k for k in sorted(eeg_data[cond][sub].keys()) if k.split('_')[0] not in ['Er', 'Sie']]
                    if mode == 'verb':
                        keys = [k for k in sorted(eeg_data[cond][sub].keys()) if k.split('_')[0] in ['Er', 'Sie']]

                # we do models last, because we don´t want to recompute rsa similarities
                for model_i, model in enumerate(sorted_models):
                    ### simple rsa
                    if args.approach == 'rsa':
                        model_data = list()
                        cog_data = list()
                        for k_i, k_one in enumerate(keys):
                            for k_two_i, k_two in enumerate(keys):
                                if k_two_i <= k_i:
                                    continue
                                key = tuple(sorted((k_one, k_two)))
                                try:
                                    model_data.append(sims[model][key])
                                except KeyError:
                                    #print(key)
                                    continue
                                cog_data.append(sub_data[key][t_i])
                        corr = scipy.stats.spearmanr(
                                                     model_data, 
                                                     cog_data,
                                                     ).statistic
                        res[model_i, sub_i, t_i, iteration] = corr
                    else:
                        ### training 80%, testing 20%
                        test_size = int(len(keys)*0.2)
                        test_keys = random.sample(keys, k=test_size)
                        train_keys = [k for k in keys if k not in test_keys]
                        ### rsa encoding
                        if args.approach == 'rsa_encoding':
                            for test_item in test_keys:
                                pred = list()
                                for k in train_keys:
                                    ### we don´t consider cases where the name was the same
                                    if k.split('_')[1] == test_item.split('_')[1]:
                                        continue
                                    curr_key = tuple(sorted((test_item, k)))
                                    eeg_sim = sims[model][curr_key]*sub_data[k][elecs, :][:, t_idxs]
                                    pred.append(eeg_sim)
                                pred = numpy.average(pred, axis=0)
                                corr = scipy.stats.spearmanr(
                                                    sub_data[test_item][elecs, :][:, t_idxs], 
                                                    pred,
                                                    ).statistic
                                res[model_i, sub_i, t_i, iteration] = corr
                        else:
                            raise RuntimeError()
                            model_data = models[model]
                            sub_corr = list()
                            for t in range(len(eeg.times)):
                                t_corrs = list()
                                for _ in range(50):
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

    pkl_res = {
                'models' : sorted_models, 
                'times' : [t[0] for t in time_ranges],
                'results' : res,
                }
                
    with open(os.path.join(curr_out, '{}_{}_encoding.pkl'.format(cond, mode)), 'wb') as o:
        pickle.dump(pkl_res, o)

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
                             'all_only', 
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
                          'debug_{}'.format(args.debug).lower(),
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
if args.locations == 'all_only':
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

### multiprocessing
with multiprocessing.Pool() as p:
    p.map(process_case, inputs)
