#!/usr/bin/env python
# -*- coding: utf-8 -*-


def create_key(template, outtype=('nii.gz',), annotation_classes=None):
    if template is None or not template:
        raise ValueError('Template must be a valid format string')
    return template, outtype, annotation_classes


def infotodict(seqinfo):

    # paths in BIDS format
    anat = create_key('sub-{subject}/anat/sub-{subject}_T1w_orig')
    info = {anat: []}

    for s in seqinfo:

        if ('t1' in s.series_description):
            info[anat].append({'item': s.series_id})

    return info
