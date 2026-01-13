#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
script to run mne-based maxfilter on the data.

Here, we will use MNE to compute the empty room data. This has the following
advantage:
    - Automatic bad and flat channel detection,
    - Application of eSSS in addition to tSSS
    - more robust against head movement offsets

"""

import os
import argparse
import mne
import misc
from mne.preprocessing import find_bad_channels_maxwell
import pandas as pd
import joblib

mne.set_config("MNE_USE_NUMBA", "true")
mne.set_config("MNE_USE_CUDA", "true")

raw_meg_files_folder = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-raw/data-MEG/'
raw_empty_rooms = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-raw/data-empty-room/'

fine_cal_file = os.path.abspath("./calibration_files/sss_cal.dat")
crosstalk_file = os.path.abspath("./calibration_files/ct_sparse.fif")

assert os.path.isfile(fine_cal_file)
assert os.path.isfile(crosstalk_file)

mem = joblib.memory.Memory('./joblib-cache/')
find_bad_channels_maxwell_cached = mem.cache(find_bad_channels_maxwell)

def run_maxfiltering(subject):
    assert isinstance(subject, int)

    folders = misc.list_files(
        f"{raw_meg_files_folder}/mfr_{subject:02d}/",
        only_folders=True
    )
    assert len(folders) == 1, f"more than one folder for subject {subject}: {folders}"

    files = {
        "main": misc.list_files(folders[0], patterns="*main.fif"),
        "rest1": misc.list_files(folders[0], patterns="*rs1.fif"),
        "rest2": misc.list_files(folders[0], patterns="*rs2.fif"),
    }

    df = pd.DataFrame()

    raws = {}
    headpos = {}

    for task, file in files.items():
        assert len(file) == 1
        raw = mne.io.read_raw(file[0], preload=True)

        auto_noisy_chs, auto_flat_chs = find_bad_channels_maxwell_cached(
            raw,
            cross_talk=crosstalk_file,
            calibration=fine_cal_file,
            # return_scores=True,
            verbose=True,
        )

        raw.info["bads"] = list(set(
            raw.info["bads"] + auto_noisy_chs + auto_flat_chs
        ))

        df = pd.concat([df,
                        pd.DataFrame({'subject': f'{subject:02d}',
                                      'task': task,
                                      'noisy': ",".join(auto_noisy_chs),
                                      'flat': ",".join(auto_flat_chs),
                                      }, index=[0])], ignore_index=True)


        headpos_file = file[0][:-4] + "_headpos.fif"

        if os.path.isfile(headpos_file):
            head_pos = mne.chpi.read_head_pos(headpos_file)
        else:
            chpi_amplitudes = mne.chpi.compute_chpi_amplitudes(raw)
            chpi_locs = mne.chpi.compute_chpi_locs(raw.info, chpi_amplitudes)
            head_pos = mne.chpi.compute_head_pos(raw.info, chpi_locs, verbose=True)
            mne.chpi.write_head_pos(headpos_file, head_pos)

        # save for later
        raws[task] = raw
        headpos[task] = head_pos

    df.to_csv(folders[0] + '/bad_chs.csv', index=False)


    # --- reference transform from main task ---
    dest = mne.preprocessing.compute_average_dev_head_t(
                            raws["main"], headpos["main"]
                        )

    # prepare empty room recording projection vectors for eSSS
    raw_er = misc.get_closest_raw(raw, raw_empty_rooms)
    extended_proj = mne.compute_proj_raw(
                        raw_er.pick('meg'),
                        n_grad=3,
                        n_mag=3,
                        meg="combined",
                    )

    # --- second pass: Maxwell filtering with tSSS and eSSS---
    for task, raw in raws.items():
        out = str(raw.filenames[0]).replace(".fif", "_trans[main]_etsss_mc.fif")
        if os.path.isfile(out):
            print(f'{subject}-{task} already computed! skip.')
            continue
        raw_sss = mne.preprocessing.maxwell_filter(
            raw,
            calibration=fine_cal_file,
            cross_talk=crosstalk_file,
            st_duration=10.0,          # tSSS
            head_pos=headpos[task],
            extended_proj=extended_proj,
            destination=dest,          # rotate to main mean head position
            coord_frame="head",
            verbose=True,
        )

        raw_sss.save(out, overwrite=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject",
        type=int,
        help="Subject number (e.g. 1 or 01)"
    )
    args = parser.parse_args()

    if args.subject:
        run_maxfiltering(args.subject)
    else:
        print('No --subject supplied, will run for all 30 participants')
        for i in range(1, 31):
            run_maxfiltering(i)
