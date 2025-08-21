import mne
import numpy
import os

errors = set()
#folder = os.path.join('surpreegtms', 'sourcedata')
folder = os.path.join('surpreeg', 'sourcedata')
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
        eeg.filter(
                   #l_freq=0.5, 
                   l_freq=0.1,
                   h_freq=30,
                   )
        raw_events = mne.events_from_annotations(eeg, verbose=False)[0]
        good_events = [l for l in raw_events if l[-1] in triggers]
        try:
            assert len(good_events) == len(tsv_events)
        except AssertionError:
            errors.add(f)
        '''
            if len(good_events) < len(tsv_events):
                real_tsv = [tsv_events[0]]
                counter = 0
                for _ in range(1, len(good_events)):
                    if good_events[_][-1] == float(tsv_events[_+counter][header.index('trigger')]):
                        real_tsv.append(tsv_events[_+counter])
                    else:
                        counter += 1
                import pdb; pdb.set_trace()

        ### correcting errors
        try:
            assert len(good_events) == len(tsv_events)
        except AssertionError:
            assert str(good_events[0][-1])[-1] == '2'
            corr_events = list()
            starting_sample = good_events[0][0]
            starting_t = float(tsv_events[0][header.index('onset')])
            ### first event
            corr_events.append(good_events[0])
            ### following events
            for ev in tsv_events[1:]:
                curr_t = float(ev[header.index('onset')])-starting_t
                sample = starting_sample+(curr_t*2000)
                corr_events.append([sample, 0, int(ev[header.index('trigger')])])
            corr_events = numpy.array(corr_events, dtype=numpy.int32)
            import pdb; pdb.set_trace()
        '''
        epochs = mne.Epochs(
                            eeg, 
                            events=raw_events, 
                            event_id=list(triggers), 
                            tmin=-.4,
                            #tmax=1.2, 
                            tmax=2., 
                            baseline=None, 
                            picks='eeg',
                            detrend=1.
                            )
        short_epochs = mne.Epochs(
                                  eeg, 
                                  events=raw_events, 
                                  event_id=list(triggers), 
                                  tmin=-.5, tmax=-0.05, 
                                  baseline=None, 
                                  picks='eeg',
                                  detrend=1.,
                                  )
        epochs.decimate(5)
        '''
        short_epochs.decimate(5)
        model = mne.decoding.Scaler(
                                    epochs.info, 
                                    scalings='median',
                                    ).fit(short_epochs.get_data())
        z_scored = model.transform(
                                   epochs.get_data(),
                                   )
        array_ep = mne.EpochsArray(
                                   z_scored, 
                                   epochs.info, 
                                   events=epochs.events, 
                                   tmin=-.4,
                                   )
        '''
        array_ep = epochs.copy()
        array_ep.save(
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
