#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 09:37:28 2024

@author: simon.kern
"""
import os
import mne
from pathlib import Path
from tqdm import tqdm
import mne_bids
import events_conversion
import shutil
from mne_bids import BIDSPath, write_raw_bids, write_anat
from mne_bids import write_meg_crosstalk, write_meg_calibration
import misc
import stimer
from stimer import ContextProfiler
# change this to the raw file folder
raw_files_folder = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-raw/'

bids_root_path = os.path.abspath(os.path.dirname(vars().get('__file__', '')) + '/../')
# bids_root = BIDSPath(root=bids_root_path)


subjects = [x for x in os.listdir(f'{raw_files_folder}/data-MEG/') if x.startswith('mfr')]
print(f'{len(subjects)=} subjects found')
# asd
#%% convert to BIDS
reports = []
for subj in tqdm(sorted(subjects), desc='processing subjects'):
# with ContextProfiler():
    subj_id = subj.split('_', 1)[-1]
    assert len(subj_id)==2


    fif_folders = misc.list_files(
        f'{raw_files_folder}/data-MEG/{subj}/',
        only_folders=True
    )
    assert len(fif_folders) == 1, f"fif_folders>1 for subject {subj}: {fif_folders}"

    fif_folder = fif_folders[0]

    files_subj = [x for x in os.listdir(fif_folder) if x.endswith('mc.fif')]

    #### 1) first convert the two resting states
    # bids_path = BIDSPath(subject=subj_id, root=bids_root)
    for rs in [1, 2]:
        fif_file = [x for x in files_subj if f'_rs{rs}_' in x]
        assert len(fif_file)==1
        raw = mne.io.read_raw_fif(f'{fif_folder}/{fif_file[0]}')
        assert misc.check_maxfilter(raw)=='mne-tsss'
        [h for h in raw.info['proc_history']]
        raw, report = misc.check_and_fix_channels(raw)
        events = mne.find_events(raw, min_duration=3/raw.info['sfreq'])
        event_id = {'resting state start': 1, 'resting state stop': 2}
        bids_task = BIDSPath(subject=subj_id,
                             datatype='meg',
                             task=f'rest{rs}',
                             root=bids_root_path)

        write_raw_bids(raw=raw,
                       allow_preload = True,
                       bids_path=bids_task,
                       events=events,
                       event_id=event_id,
                       format='FIF',
                       # empty_room=raw_er,
                       overwrite=True,
                       verbose=True
                       )
        reports += [report]

    #### 2) next convert the main data
    fif_file = [x for x in files_subj if f'_main_' in x and x.endswith('.fif') and 'tsss' in x and not '-1.' in x]
    assert len(fif_file)==1
    raw = mne.io.read_raw_fif(f'{fif_folder}/{fif_file[0]}')
    assert misc.check_maxfilter(raw)=='mne-tsss'

    raw, report = misc.check_and_fix_channels(raw)
    reports += [report]

    events = mne.find_events(raw, min_duration=3/raw.info['sfreq'])
    stimuli = ['Face', 'House', 'Cat', 'Shoe', 'Chair']
    event_id = ({'Break start': 91,
                'Break stop': 92,
                'Sequence Buffer start': 61,
                'Sequence Buffer stop': 62,
                'sequence sound error': 70,
                'sequencesound coin': 71,
                'localizer sound error': 30,
                'localizer sound coin': 31,
                'fixation pre 1': 81,
                'fixation pre 2': 82,
                } | {f'localizer {stim} onset':i for i, stim in enumerate(stimuli, 1)}
                  | {f'localizer {stim} distractor onset':i+100 for i, stim in enumerate(stimuli, 1)}
                  | {f'cue {stim}':i+10 for i, stim in enumerate(stimuli, 1)}
                  | {f'sequence {stim} onset':i+20 for i, stim in enumerate(stimuli, 1)}
                )

    bids_task_main= BIDSPath(subject=subj_id,
                             datatype='meg',
                             task=f'main',
                             root=bids_root_path)
    write_raw_bids(raw=raw,
                    bids_path=bids_task_main,
                    allow_preload = True,
                    events=events,
                    event_id=event_id,
                    format='FIF',
                    # empty_room=raw_er,
                    overwrite=True,
                    verbose=True
                   )

    #### 3) crosstalk and calibration files
    bids_task_cal = BIDSPath(subject=subj_id,
                             datatype='meg',
                             root=bids_root_path)
    cal_fname = os.path.join(raw_files_folder, "sss_cal.dat")
    ct_fname = os.path.join(raw_files_folder, "ct_sparse.fif")

    write_meg_calibration(cal_fname, bids_task_cal)
    write_meg_crosstalk(ct_fname, bids_task_cal)

    #### 4) behavioural data
    # basically sourdedata is a fractal BIDS folder
    bids_task_source = BIDSPath(subject=subj_id,
                         datatype='beh',
                         task=f'main',
                         root=bids_root_path + '/sourcedata/')
    bids_task_source.mkdir()

    # filter CSV file
    subj_folder = f'{raw_files_folder}/data-logs/{subj.replace("_", "-").upper()}/'
    files_subj = [x for x in os.listdir(subj_folder) if x.endswith('csv') ]
    files_subj = [x for x in files_subj if (('main_' in x) & x.startswith(f'{int(subj_id):02d}'))]
    files_subj = [f'{subj_folder}/{x}' for x in files_subj]
    assert len(files_subj)==1
    csv_file = files_subj[0]
    log_file = files_subj[0][:-3] + 'log'
    shutil.copy(log_file, str(bids_task_source.fpath) + '.log')

    bids_task_main.update(datatype='beh', suffix='beh')
    bids_task_main.mkdir()
    df_subj = events_conversion.convert_psychopy_to_bids(csv_file)
    df_subj['subject'] = f'sub-{subj_id}'
    df_subj['session'] = 1

    df_subj.to_csv(str(bids_task_main.fpath) + '.tsv', sep='\t', index=False,
                   na_rep='NaN')

    # asd
    #### 5) MRI data
    ### MRI DATA IS ALREADY CONVERTED VIA HEUDICONV
    # t1w_path = BIDSPath(subject=subj_id,
    #                     datatype='anat',
    #                     # task=f'T1w',
    #                     root=bids_root_path)

    # t1w_bids_path = write_anat(
    #     image='/data/fastreplay/Fast-Replay-MEG-bids/sub-01/anat/sub-02_T1w.nii.gz',  # path to the MRI scan
    #     bids_path=t1w_path,
    #     landmarks=None,
    #     deface=False,
    #     overwrite=True,
    #     verbose=True,  # this will print out the sidecar file
    # )

#%%  copy emptyroom data
empty_fifs = os.listdir(raw_files_folder + '/data-empty-room/')
for fif_file in tqdm(empty_fifs, desc='writing empty rooms'):
    raw = mne.io.read_raw(raw_files_folder + '/data-empty-room/' +  fif_file, verbose='ERROR')
    er_date = fif_file.split('_')[1]
    er_bids_path = BIDSPath(
        subject="emptyroom", session=er_date, task="noise", root=bids_root_path
    )
    write_raw_bids(raw, er_bids_path, overwrite=True)
