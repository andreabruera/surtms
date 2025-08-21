import numpy
import os
import scipy

folder = '/data/pt_03070/MorphSem2_LLM/eeg-preprocessing/rawdata/EEG/'
fnames = dict()
for f in os.listdir(folder):
    if '.eeg' not in f:
        continue
    subject = int(f.split('_')[0][2:])
    try:
        fnames[subject].append(f.split('.')[0])
    except KeyError:
        fnames[subject] = [f.split('.')[0]]

pos = dict()
for s, fs in fnames.items():
    for ses in [1, 2]:
        ## make montage
        montage_file = os.path.join(
                                    'resources',
                                    'electrode_positions', 
                                    'MorphSem2_vp{:02}_s{}.csv'.format(
                                                                       s, 
                                                                       ses)
                                    )
        with open(montage_file) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split(',')
                if 'VUNTER' in line:
                    elec = 'EOGV'
                elif 'VOBEN' in line:
                    elec = 'EOGH'
                elif 'HL' in line:
                    continue
                elif 'HR' in line:
                    continue
                else:
                    elec = line[1].replace('Z', 'z').replace('FP', 'Fp')
                try:
                    pos[elec].append(numpy.array(
                                        line[2:], 
                                        dtype=numpy.float32,
                                            ))
                except KeyError:
                    pos[elec] = [numpy.array(
                                        line[2:], 
                                        dtype=numpy.float32,
                                            )]
pos = {k : numpy.nanmean(v, axis=0) for k, v in pos.items()}
for k, v in pos.items():
    #print([k, v])
    assert 'nan' not in [str(_) for _ in v]
clusters = dict()
for k_i, k in enumerate(pos):
    if k in ['A1', 'A2', 'EOGV', 'EOGH',]:
        continue
    if k not in clusters.keys():
        clusters[k] = dict()
    for k_two_i, k_two in enumerate(pos):
        if k_two in ['A1', 'A2', 'EOGV', 'EOGH',]:
            continue
        if k_two not in clusters.keys():
            clusters[k_two] = dict()
        if k_i <= k_two_i:
            continue
        distance = scipy.spatial.distance.euclidean(
                                                    pos[k], 
                                                    pos[k_two],
                                                    )
        clusters[k][k_two] = distance
        clusters[k_two][k] = distance
os.makedirs('resources', exist_ok=True)
with open(
          os.path.join(
                       'resources',
                       'avg_electrode_distances.tsv',
                       ),
              'w') as o:
    o.write('cluster_center\tother_electrode, euclidean_distance**\n')
    for start, other_dict in clusters.items():
        o.write('{}\t'.format(start))
        sort = sorted(other_dict.items(), key=lambda item:item[1])
        for sort_elec, dist in sort:
            o.write('{},{}\t'.format(sort_elec, dist))
        o.write('\n')
