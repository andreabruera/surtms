import matplotlib
import mne
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

font_setup('../../../fonts')

for root, direc, fz in os.walk('results'):
    with tqdm() as counter:
        for f in fz:
            if '.pkl' not in f:
                continue
            locations = root.split('/')[-4]
            if locations == 'searchlight':
                continue
            subjects = root.split('/')[-6]
            approach = root.split('/')[-7]
            with open(os.path.join(root, f), 'rb') as i:
                res = pickle.load(i)
            ### building curr_res
            curr_res = {m : res['results'][m_i] for m_i, m in enumerate(res['models'])}
            times = res['times']
            cond = f.split('_')[-3]
            mode = f.split('_')[-2]
            fig, ax = pyplot.subplots(
                                      constrained_layout=True,
                                      figsize=(20, 10),
                                      )
            ax.hlines(y=0., xmin=min(times), xmax=max(times), color='black')
            ax.vlines(x=0., ymin=-.2, ymax=.2, color='black')
            raw_ps = list()
            colours = {
                       'comp' : 'black',
                       'Phonological similarity' : 'salmon',
                       'Word surprisal (GPT2)' : 'teal',
                       'Semantic similarity (GPT2)' : 'orange',
                       'gpt2-small_surprisal' : 'goldenrod',
                       'fasttext' : 'gray',
                       }
            ys = {
                       'comp' : -.07,
                       'Phonological similarity' : -.07,
                       'Word surprisal (GPT2)' : -.08,
                       'Semantic similarity (GPT2)' : -.09,
                       'gpt2-small_8' : -.06,
                       'fasttext' : -0.05,
                       }
            for model, model_res in curr_res.items():
                if 'leven' in model:
                    model = 'Phonological similarity'
                elif 'gpt2' in model:
                    if 'surp' in model:
                        model = 'Word surprisal (GPT2)'
                    else:
                        model = 'Semantic similarity (GPT2)'
                else:
                    continue
                if 'across' in subjects:
                    avg_plot = numpy.nanmean(model_res[0, :, :], axis=1)
                    std_plot = numpy.nanstd(model_res[0, :, :], axis=1)
                    p_plot = model_res[0]
                else:
                    if approach == 'rsa':
                        avg_plot = numpy.nanmean(numpy.nanmean(model_res[:, :, :], axis=0), axis=1)
                        std_plot = numpy.nanstd(numpy.nanmean(model_res[:, :, :], axis=0), axis=1)
                        #p_plot = model_res[0]
                        p_plot = numpy.nanmean(model_res, axis=0)
                    else:
                        avg_plot = numpy.nanmean(numpy.nanmean(model_res, axis=2), axis=0)
                        #std_plot = numpy.std(numpy.average(model_res, axis=2), axis=0)
                        std_plot = scipy.stats.sem(numpy.nanmean(model_res, axis=2), axis=0, nan_policy='omit')
                        p_plot = numpy.nanmean(model_res, axis=2)
                ax.plot(
                        times, 
                        avg_plot, 
                        label=model,
                        color=colours[model],
                        )
                ax.fill_between(
                                times, 
                                avg_plot-std_plot,
                                avg_plot+std_plot,
                                alpha=0.2,
                                color=colours[model],
                                )
                if approach == 'rsa':
                    raw_ps = list()
                    for iter_vals in p_plot:
                        assert len(iter_vals) == 1000
                        one_side = (sum([1 for _ in iter_vals if _<0.])+1)/1001
                        #two_side = (sum([1 for _, __ in zip(one, two) if _-__>0.])+1)/1001
                        #p = min(one_side, two_side)*2
                        assert one_side>=0. and one_side<=1.
                        #print(p)
                        raw_ps.append(one_side)
                else:
                    '''
                    threshold_tfce = dict(start=0, step=0.2)
                    t, _, ps, __ = mne.stats.permutation_cluster_1samp_test(
                                                        p_plot, 
                                                        threshold=threshold_tfce, 
                                                        adjacency=None,
                                                        n_jobs=-1,
                                                        tail=1.,
                                                        )
                    '''
                    raw_ps = list()
                    for t_i in tqdm(range(p_plot.shape[-1])):
                        t_res = p_plot[:, t_i]
                        real_avg = numpy.average(t_res)
                        false = list()
                        for _ in range(1000):
                            false_res = t_res*numpy.array(random.choices([-1, 1], k=t_res.shape[0]))
                            false_avg = numpy.average(false_res)
                            false.append(false_avg)

                        one_side = (sum([1 for _ in false if _>real_avg])+1)/1001
                        assert one_side>=0. and one_side<=1.
                        raw_ps.append(one_side)
                ps = scipy.stats.false_discovery_control(raw_ps)
                for p_i, p in enumerate(ps):
                    if p<0.05:
                        print(p)
                        if 'across' in subjects:
                            y = ys[model]
                        else:
                            y = ys[model]*.5
                        ax.scatter(
                                  times[p_i],
                                  y,
                                    color=colours[model],
                                    )
                '''
                for r in model_res:
                    assert len(r) == 1000
                    p = (sum([1 for _ in r if _<0.])+1)/1001
                    #two = (sum([1 for _ in r if _>0.])+1)/1001
                    #p = min([one, two])*2
                    assert p>=0. and p<=1.
                    #print(p)
                    raw_ps.append((model, p))
            for t in range(len(times)):
                #model_res = numpy.array(model_res[1::2])
                #one = numpy.array(curr_res[sorted(curr_res.keys())[0]][1::2])[t]
                #two = numpy.array(curr_res[sorted(curr_res.keys())[1]][1::2])[t]
                one = numpy.array(curr_res[sorted(curr_res.keys())[0]])[t]
                two = numpy.array(curr_res[sorted(curr_res.keys())[1]])[t]
                assert len(one) == 1000
                #p = (sum([1 for _ in r if _<0.])+1)/1001
                one_side = (sum([1 for _, __ in zip(one, two) if _-__<0.])+1)/1001
                two_side = (sum([1 for _, __ in zip(one, two) if _-__>0.])+1)/1001
                p = min(one_side, two_side)*2
                #assert p>=0. and p<=1.
                #print(p)
                raw_ps.append(('comp', p))
            fdr_ps = scipy.stats.false_discovery_control([v[1] for v in raw_ps])
            plot_ps = dict()
            for _ in range(len(raw_ps)):
                fdr = fdr_ps[_]
                model = raw_ps[_][0]
                try:
                    plot_ps[model].append(fdr)
                except KeyError:
                    plot_ps[model] = [fdr]
            for model, model_ps in plot_ps.items():
                assert len(model_ps) == len(times)
                for p_i, p in enumerate(model_ps):
                    if p<0.05:
                        if model == 'comp':
                            ax.fill_between(
                           [times[max(1, p_i)-1], times[p_i]],
                           [numpy.average(curr_res[sorted(curr_res.keys())[0]][1::2], axis=1)[max(1, p_i)-1],
                           numpy.average(curr_res[sorted(curr_res.keys())[0]][1::2], axis=1)[p_i]],
                           [numpy.average(curr_res[sorted(curr_res.keys())[1]][1::2], axis=1)[max(1, p_i)-1],
                           numpy.average(curr_res[sorted(curr_res.keys())[1]][1::2], axis=1)[p_i]],
                                    color=colours[model],
                                    alpha=0.1,
                                    linewidth=0,
                                    )
                        else:
                            ax.scatter(
                                  times[p_i],
                                  ys[model],
                                    color=colours[model],
                                    )

                '''
            ax.legend(fontsize=23)
            xticks = [v*0.1 for v in range(round(min(times)*10), round(max(times)*10))]
            if 'across' in subjects:
                ax.set_ylim(bottom=-.1, top=.2)
            else:
                ax.set_ylim(bottom=-.05, top=.1)
            ax.set_xticks(xticks)
            pyplot.xticks(fontsize=18)
            pyplot.yticks(fontsize=18)
            pyplot.ylabel('Spearman rho', fontsize=21, fontweight='bold')
            pyplot.xlabel('Seconds after stimulus', fontsize=21, fontweight='bold')
            ax.set_title(
                    'Condition:   {}\nPosition:   {}'.format(cond, mode), 
                         fontsize=35, 
                         fontweight='bold',
                         )
            out_f = root.replace('results', 'plots')
            os.makedirs(out_f, exist_ok=True)
            f_name = os.path.join(out_f, f.replace('pkl', 'jpg'))
            print(f_name)
            pyplot.savefig(f_name, dpi=300)
            counter.update(1)
            pyplot.clf()
            pyplot.close()
