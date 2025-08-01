import mne
import os

triggers = {
        'ehcw' : 6,
        'elcw' : 8,
        'facw' :11,
        'fccw' : 13,
        'ehcn' : 16,
        'elcn' : 18,
        'facn' : 21,
        'fccn' : 23,
            }

files = dict()
with open('transcriptions.txt') as i:
    for l in i:
        line = l.strip().split(';')
        files['{}.wav'.format(line[0])] = line[1:]

durations = dict()
for f in os.listdir('durations'):
    with open(os.path.join('durations', f)) as i:
        for l in i:
            line = l.strip().split(';')
            durations[line[0]] = line[1:]

cond_mapper = {
        's1' : 's1',
        's2' : 's2',
        'pIFG' : 'f',
        'pSTG' : 't',
        }
        #s1 = first sham / s2 = second sham / f = pIFG / t = pSTG
out_mapper = {
        's1' : 'sham',
        's2' : 'AGcTBS',
        'pIFG' : 'pIFGrTMS',
        'pSTG' : 'pSTGrTMS',
        }
        

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

for s, fs in fnames.items():
    assert len(fs) == 6
    for f in fs:
        ses = int(f.split('_')[1][1])
        block = int(f.split('_')[2][1])
        cond = out_mapper[f.split('_')[-1]]
        out_folder = os.path.join(
                                  'surpreegtms',
                                  'sourcedata',
                                  'sub-{:02}'.format(s),
                                  'ses-{:02}'.format(ses),
                                  )
        os.makedirs(out_folder, exist_ok=True)
        out_f = 'sub-{:02}_ses-{:02}_task-lexicaldecision_acq-{}_run-{:02}'.format(
                                  s,
                                  ses,
                                  cond,
                                  block,
                                  )

        eeg = os.path.join(folder, '{}.eeg'.format(f))
        assert os.path.exists(eeg)
        vmrk = os.path.join(folder, '{}.vmrk'.format(f))
        assert os.path.exists(vmrk)
        vhdr = os.path.join(folder, '{}.vhdr'.format(f))
        assert os.path.exists(vhdr)
        '''
        data = mne.io.read_raw_brainvision(
                                           vhdr, 
                                           eog=('EOGH', 'EOGV'),
                                           misc=('O1', 'O2'),
                                           )
        #montage = 'easycap-M10'
        #data.set_montage(montage)
        data.save(os.path.join(
                               out_folder,
                               '{}_eeg.fif'.format(out_f),
                               )
                  )
        '''
        ### reading log
        cond = cond_mapper[f.split('_')[-1]]
        log_f = '{:02}_{}_b{}_{}_MorphSem2_TMSEEG.log'.format(
                                                             s,
                                                             ses,
                                                             block,
                                                             cond,
                                                             )
        full_log = os.path.join('raw_logs', log_f)
        #print(full_log)
        assert os.path.exists(full_log)
        lines = list()
        with open(full_log) as i:
            for l_i, l in enumerate(i):
                line = l.strip().split('\t')
                if l_i < 3:
                    continue
                elif l_i == 3:
                    header = line.copy()
                    assert len(header) == 13
                else:
                    if len(line) < 4:
                        #print(line)
                        continue
                    lines.append(line)
        code = header.index('Code')
        time = header.index('Time')
        dur = header.index('Duration')
        rel_codes = ['2', '5', '6', '8']
        sel_lines = list()
        missing = set()
        tr = 0
        for l in lines:
            if len(l[code]) < 3:
                continue
            if 'tms' in l[code]:
                continue
            if 'wav' in l[code]:
                wav = l[code]
                try:
                    sent = files[wav]
                    durat = durations[wav]
                except KeyError:
                    sent = ['NA', 'NA', 'NA', 'NA']
                    durat = ['NA', 'NA', 'NA', 'NA']
                #print(sent)
                continue
            if l[code][-1] in rel_codes:
                if l[code][-1] == '2':
                    tr += 1
                if 'w' in l[code] and 'NA' in sent:
                    missing.add(wav)
                w = sent[rel_codes.index(l[code][-1])]
                d = durat[rel_codes.index(l[code][-1])]
                t = int(l[time])/10000
                trig = '{}{}'.format(triggers[l[code][:-1]], l[code][-1])
                sel_lines.append((tr, w, trig, l[code], wav, t, d))
        print(missing)
        with open(os.path.join(
                               out_folder,
                               '{}_events.tsv'.format(out_f),
                               ),
                  'w') as o:
            o.write('onset\tduration\ttrial_n\tword\ttrigger\tcategory\twav_file\n')
            for tr, w, trig, cat, wav, t, d in sel_lines:
                o.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(t, d, tr, w, trig, cat, wav))
