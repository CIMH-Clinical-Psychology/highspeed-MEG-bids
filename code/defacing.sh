#!/bin/bash
# this script is called via "make defacing" in the parent dir.
PATH_PROJECT=$1
PYDEFACE_VERSION=37-2e0c2d
USER_ID=$(id -u)
GROUP_ID=$(id -g)

for FILE in ${PATH_PROJECT}/*/anat/*T1w_orig.nii.gz; do
    # get the filename:
	FILE_BASENAME="$(basename -- $FILE)"
    # get the parent directory:
	FILE_PARENT="$(dirname "$FILE")"
    FILE_OUT=$(echo "$FILE_BASENAME" | sed 's/_orig//')
    # run defacing:
    docker run --rm -u ${USER_ID}:${GROUP_ID} -v ${FILE_PARENT}:/input:rw  poldracklab/pydeface:${PYDEFACE_VERSION} \
        pydeface /input/${FILE_BASENAME} --outfile /input/${FILE_OUT} --force
    rm -f ${FILE}
    JSON_FILE="${FILE%.nii.gz}.json"
    mv "${FILE%.nii.gz}.json" "${FILE%_orig.nii.gz}.json"
done
