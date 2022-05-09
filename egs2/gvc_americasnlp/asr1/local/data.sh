#!/usr/bin/env bash

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# general configuration
stage=0       # start from 0 if you need to start from data preparation
stop_stage=100
SECONDS=0

 . utils/parse_options.sh || exit 1;

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

GVC_AMERICASNLP=/mount/arbeitsdaten/asr-4/denisopl/AmericasNLP/Kotiria

if [ -z "${GVC_AMERICASNLP}" ]; then
    log "Fill the value of 'GVC_AMERICASNLP' of db.sh"
    exit 1
fi

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log "data preparation started"

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage1: Preparing data for commonvoice"
    ### Task dependent. You have to make data the following preparation part by yourself.
    local/prepare_data.py ${GVC_AMERICASNLP}
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
