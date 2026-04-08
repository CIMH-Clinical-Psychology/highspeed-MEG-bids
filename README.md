# highspeed-MEG-bids

BIDS dataset for the MEG replication of Wittkuhn et al 2021.

## Creating the BIDS-dataset

This repository already contains the final BIDS dataset. For most purposes you can stop reading here. If you have the raw data, you can replicate the process by the following commands to convert the raw data to BIDS. The raw data have not been uploaded, so this is more for transparency and as a reminder for myself. Note: Beforehand you might need to edit the path to the raw data in your `code/config.env`



#### 1. run mne-maxfilter on the raw data

first we need to run maxfilter on the raw data. Some records had missing cHPI segments, but `mne` handles that by interpolating the segment's cHPI positions.

run it either by `python code/run_maxfilter.py` or if you have SLURM, use the `run_maxfilter.sbatch` script. This will add the `*_tsss.fif` files to your raw files directory.

#### 2. run anatomical preprocessing and defacing

sha

```bash
make venv
make install
make anat
make defacing
make bids

```

References

Appelhoff, S., Sanderson, M., Brooks, T., Vliet, M., Quentin, R., Holdgraf, C., Chaumon, M., Mikulan, E., Tavabi, K., Höchenberger, R., Welke, D., Brunner, C., Rockhill, A., Larson, E., Gramfort, A. and Jas, M. (2019). MNE-BIDS: Organizing electrophysiological data into the BIDS format and facilitating their analysis. Journal of Open Source Software 4: (1896).https://doi.org/10.21105/joss.01896

Niso, G., Gorgolewski, K. J., Bock, E., Brooks, T. L., Flandin, G., Gramfort, A., Henson, R. N., Jas, M., Litvak, V., Moreau, J., Oostenveld, R., Schoffelen, J., Tadel, F., Wexler, J., Baillet, S. (2018). MEG-BIDS, the brain imaging data structure extended to magnetoencephalography. Scientific Data, 5, 180110.https://doi.org/10.1038/sdata.2018.110
