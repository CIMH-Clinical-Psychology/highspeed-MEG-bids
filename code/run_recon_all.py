#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 15:56:57 2024

@author: simon.kern
"""
import os
import time
import subprocess
from subprocess import PIPE
from tqdm import tqdm
from joblib import Parallel, delayed

def recon_all(folder):
    MRI_FOLDER = folder
    SUBJ = os.path.basename(MRI_FOLDER)

    if os.path.exists(f'{subj_dir}/{SUBJ}/surf/lh.white') and \
        os.path.exists(f'{subj_dir}/{SUBJ}/surf/rh.white'):
        print(f'already reconstructed for {SUBJ}')
        return

    os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
    MRI_FOLDER = MRI_FOLDER.replace('//', '/')
    assert os.path.isdir(MRI_FOLDER)

    NIFTI_FILE = f'{MRI_FOLDER}/T1.nii.gz'

    # first convert the DICOM to NIFGI
    if not os.path.exists(NIFTI_FILE):
        convert_cmd = f'./dcm2niix -d n {MRI_FOLDER}'
        process = subprocess.Popen(convert_cmd, stdout=PIPE, stderr=PIPE, shell=True)
        while process.stdout.readable():
            line = process.stdout.readline()
            if not line: break
            print(line.decode().strip())
            time.sleep(0.05)
        t1_files = list(filter(lambda x:(x.endswith('gz') and 't1' in x.lower()), os.listdir(MRI_FOLDER)))
        if len(t1_files)==0: return
        os.rename(f'{MRI_FOLDER}/{t1_files[0]}', NIFTI_FILE)
    try:
        # now extract the regions
        recon_cmd = f'nice -n 15 recon-all -i {NIFTI_FILE} -s {SUBJ}  -all'
        process = subprocess.Popen(recon_cmd, stdout=PIPE, stderr=PIPE, shell=True)
        lines = [f'{SUBJ}: ']
        while process.stdout.readable():
            line = process.stdout.readline().decode()
            if not line: break
            lines.append(line)
            print(line.strip())
    except Exception as e:
        print('#'*40, MRI_FOLDER)
        import traceback
        print(traceback.format_exc())
        return f'{SUBJ} : Error {traceback.format_exc()}'

    return '\n'.join(lines)

project_dir = '/zi/flstorage/group_klips/data/data/Fast-Replay-MEG/'
subj_dir = f'{project_dir}/freesurfer/'
folders = [f'{project_dir}/data-MRI/{x}' for x in os.listdir(project_dir + 'data-MRI')]
folders = [f for f in folders if os.path.isdir(f)]
FS_HOME = os.environ.get('FREESURFER_HOME')
os.makedirs(subj_dir, exist_ok=True)
os.environ['SUBJECTS_DIR'] = subj_dir


assert FS_HOME, '$FREESURFER_HOME not found in env'
assert os.path.isfile(FS_HOME + '/bin/recon-all'), 'recon-all not found'
assert FS_HOME in os.environ['PATH'], 'freesurfer not on $PATH'

errs_recon = Parallel(30, backend='threading')(delayed(recon_all)(folder) for folder in tqdm(folders))
