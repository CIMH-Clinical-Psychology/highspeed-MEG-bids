#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 09:38:42 2026

@author: simon.kern
"""

import pandas as pd
import misc
raw_meg_files_folder = '/zi/flstorage/group_klips/data/data/Simon/highspeed/highspeed-MEG-raw/data-MEG/'


files = misc.list_files(raw_meg_files_folder, patterns='bad_chs.csv',recursive=True)

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

df.to_csv(raw_meg_files_folder + '/bad_channels.csv', index=False)
