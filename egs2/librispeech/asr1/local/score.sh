#!/usr/bin/env bash

python=python3

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -lt 1 ]; then
  echo "Usage: local/score.sh <asr-exp-dir>"
  exit 1;
fi

asr_expdir=$1

for text in ${asr_expdir}/*/*/text; do
	_dir=$(dirname ${text})
	_scoredir=${_dir}/score_per
	dset=$(basename ${_dir})

	if [[ ${dset} == *_phn ]]; then
		continue
	else
		dset=${dset}_phn
	fi

	_data=dump_libritts/raw/${dset}

	mkdir -p ${_dir}/score_per

	paste \
		<(<"${_data}/text" \
		      ${python} -m espnet2.bin.tokenize_text  \
			  -f 2- --input - --output - \
			  --token_type char \
			  ) \
		<(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
		    >"${_scoredir}/ref.trn"

	local/phonemize_text.py ${text} ${text}.phn

	paste \
		<(<"${text}.phn" \
		      ${python} -m espnet2.bin.tokenize_text  \
			  -f 2- --input - --output - \
			  --token_type char \
			  ) \
		<(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
		    >"${_scoredir}/hyp.trn"

	sclite \
            -r "${_scoredir}/ref.trn" trn \
            -h "${_scoredir}/hyp.trn" trn \
            -i rm -o all stdout > "${_scoredir}/result.txt"

            echo "Write PER result in ${_scoredir}/result.txt"
            grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"

done

exit 0
