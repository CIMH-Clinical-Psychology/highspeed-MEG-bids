#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 11:43:48 2025

@author: simon.kern
"""
import os
from functools import cache
import numpy as np
import mne
from joblib import load as _joblib_load
from pathlib import Path
import datetime

joblib_load = cache(_joblib_load)

def make_maxfilter_filename(filename, method='tsss', trans=None, mc=True):
    """return a valid maxfilter file name, ignoring mne conventions"""
    assert method in ['tsss', 'sss', 'etsss', 'esss']
    file, ext = os.path.splitext(filename)

    maxfilter_name = file + (f'_trans[{trans}]'  if trans else '')
    maxfilter_name += f'_{method}'
    maxfilter_name += '_mc' if mc else ''
    maxfilter_name += ext

    return maxfilter_name

def check_and_fix_channels(raw):
    """check for missing channels or empty channels or NaN channels"""
    report = {'filename': os.path.basename(raw._filenames[0]),
              'missing': []}
    template_info = mne.io.read_info('template-info.fif')

    ch_types = {'BIO001': 'bio',
                'BIO002': 'bio',
                'BIO003': 'bio',
                'MEG2112': 'grad',
                'MEG2211': 'mag',}

    # check for missing channels
    missing = set(template_info.ch_names) - set(raw.ch_names)
    chs_add = []
    bads = raw.info['bads'].copy()

    raw.load_data()
    empty_ch = np.zeros_like(raw.get_data(0))
    for ch in missing:
        report['missing'] += [ch]
        if ch.startswith('CHPI'):
            # this just means motion correction didn't run - possibly
            # this file is just fine as it is.
            continue
        elif ch in ch_types:
            info = mne.create_info(ch_names=[ch], sfreq=raw.info['sfreq'], ch_types=ch_types[ch])

            ch_idx = template_info.ch_names.index(ch)
            ch_t = template_info['chs'][ch_idx]
            for key in ('loc', 'coil_type', 'kind', 'unit', 'unit_mul', 'coord_frame'):
                info['chs'][0][key] = ch_t[key]

            raw_new = mne.io.RawArray(data=empty_ch, info=info,
                                      first_samp=raw.first_samp, verbose='WARNING')
            chs_add += [raw_new]

            # these channels can be interpolated, unlike the BIO channels
            if ch.startswith('MEG'):
                bads += [ch]
            print(f'  -Adding {ch}')

        else:
            raise Exception(f'{ch=} missing! don\'t know what to do')

    if chs_add:
        raw.load_data(verbose='WARNING')
        raw.add_channels(chs_add, force_update_info=True)
        if bads:
            raw.info['bads'] += bads
            raw.interpolate_bads()


    # next check if any data is nan
    data = raw.get_data()
    for ch, d in zip(raw.ch_names, data):
        if 'CHPI' in ch:
            continue
        assert not np.isnan(d).any(), f'some data is NaN for {ch=}!'
    return raw, report


def get_closest_raw(raw, folder_of_fifs):
    """get the recording that is closest to the reference raw recording date

    this function finds you an empty room recording that is closest"""

    date = raw.info['meas_date']
    assert isinstance(date, datetime.datetime)

    files = list_files(folder_of_fifs, patterns='*.fif')
    raws = [mne.io.read_raw(f, preload=False, verbose='ERROR') for f in files]
    deltas = [abs(r.info['meas_date']-date) for r in raws]
    idx_min = np.argmin(deltas)
    return raws[idx_min]

def check_maxfilter(raw):
    """for a raw, check that maxfiltering was done by MNE and with tSSS"""
    for proc in raw.info['proc_history']:
        if 'max_info' in proc:
            creator = 'mne' if 'mne' in proc['creator'] else proc['creator']
            tsss = 'tsss' if proc['max_info'].get('max_st') is not None else 'sss'
            return f'{creator}-{tsss}'
    return 'not-maxfiltered'

def list_files(path, exts=None, patterns=None, relative=False, recursive=False,
               subfolders=None, only_folders=False, max_results=None,
               case_sensitive=False):
    """
    will make a list of all files with extention exts (list)
    found in the path and possibly all subfolders and return
    a list of all files matching this pattern

    :param path:  location to find the files
    :type  path:  str
    :param exts:  extension of the files (e.g. .jpg, .jpg or .png, png)
                  Will be turned into a pattern internally
    :type  exts:  list or str
    :param pattern: A pattern that is supported by pathlib.Path,
                  e.g. '*.txt', '**\rfc_*.clf'
    :type:        str
    :param fullpath:  give the filenames with path
    :type  fullpath:  bool
    :param subfolders
    :param return_strings: return strings, else returns Path objects
    :return:      list of file names
    :type:        list of str
    """
    from natsort import natsort_key

    def insensitive_glob(pattern):
        f = lambda c: '[%s%s]' % (c.lower(), c.upper()) if c.isalpha() else c
        return ''.join(map(f, pattern))

    if subfolders is not None:
        import warnings
        warnings.warn("`subfolders` is deprecated, use `recursive=` instead", DeprecationWarning)
        recursive = subfolders

    if isinstance(exts, str): exts = [exts]
    if isinstance(patterns, str): patterns = [patterns]

    p = Path(path)
    assert p.exists(), f'Path {path} does not exist'
    if patterns is None: patterns = []
    if exts is None: exts = []

    if not patterns and not exts:
        patterns = ['*']

    for ext in exts:
        ext = ext.replace('*', '')
        pattern = '*' + ext
        patterns.append(pattern.lower())

    # if recursiveness is asked, prepend the double asterix to each pattern
    if recursive: patterns = ['**/' + pattern for pattern in patterns]

    # collect files for each pattern
    files = []
    fcount = 0
    for pattern in patterns:
        if not case_sensitive:
            pattern = insensitive_glob(pattern)
        for filename in p.glob(pattern):
            if filename.is_file() and filename not in files:
                if only_folders:
                    continue
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results<=fcount:
                    break
            elif filename.is_dir() and only_folders and filename not in files:
                files.append(filename)
                fcount += 1
                if max_results is not None and max_results<=fcount:
                    break


    # turn path into relative or absolute paths
    if relative:
        files = [file.relative_to(p) for file in files]

    # by default: return strings instead of Path objects
    files = [str(file) for file in files]
    files = set(files)  # filter duplicates
    return sorted(files, key=natsort_key)

if __name__=='__main__':
    import mne
    raw = mne.io.read_raw('/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-raw/data-MEG/mfr_03/MFR03_main_trans[MFR03_main]_tsss_mc.fif', preload=True)
    check_and_fix_channels(raw)
