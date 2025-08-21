import mne
import numpy
import os

errors = set()
folder = os.path.join('..', 'surpreegtms', 'sourcedata')
for root, direc, fz in os.walk(folder):
    for f in fz:
        if 'fif' not in f:
            continue
        out_fold = root.replace('sourcedata', 'derivatives')
        os.makedirs(out_fold, exist_ok=True)
        ev_f = f.replace('eeg.fif', 'events.tsv')
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
        eeg = mne.io.read_raw_fif(os.path.join(root, f), preload=True)
        ### filtering raw signal between 0.5 and 30
        eeg.filter(
                   l_freq=0.5, 
                   h_freq=30,
                   )
        raw_events = mne.events_from_annotations(eeg, verbose=False)[0]
        good_events = [l for l in raw_events if l[-1] in triggers]
        try:
            assert len(good_events) == len(tsv_events)
        except AssertionError:
            errors.add(f)
        ### keeping long epochs, just in case...
        epochs = mne.Epochs(
                            eeg, 
                            events=raw_events, 
                            event_id=list(triggers), 
                            tmin=-.5,
                            tmax=2., 
                            baseline=None, 
                            picks='eeg',
                            detrend=1.
                            )
        ### 100hz
        epochs.decimate(20)
        epochs.save(
                  os.path.join(
                                out_fold, 
                                f.replace('eeg.fif',  'eeg-epo.fif')
                                ),
                  overwrite=True,
                  )
        with open(os.path.join(out_fold, ev_f), 'w') as o:
            o.write('\t'.join(header))
            o.write('\n')
            for l in tsv_events:
                o.write('\t'.join(l))
                o.write('\n')
print(errors)
