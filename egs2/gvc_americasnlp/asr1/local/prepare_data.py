#!/usr/bin/env python3

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import glob
import os
import re
import subprocess
import sys

idir = sys.argv[1]

wavs = {}

for trs in glob.glob(f"{idir}/**/*.trs", recursive=True):
    dir_name = trs[: trs.rindex("/")]
    wav_list = glob.glob(f"{dir_name}/*.wav") + glob.glob(f"{dir_name}/*.WAV")
    recoid = trs[trs.rindex("/") + 1 : -4]
    wavs[recoid] = wav_list[0]

punct_re = re.compile('[.!"()*,-;=?\[\]]')

for subset in ["train", "dev", "test"]:
    odir = "data/" + subset
    os.makedirs(odir, exist_ok=True)
    with open(f"{idir}/kotiria_{subset}.csv", encoding="utf-8") as metafile, open(
        odir + "/text", "w", encoding="utf-8"
    ) as text, open(odir + "/wav.scp", "w") as wavscp, open(
        odir + "/utt2spk", "w"
    ) as utt2spk:
        meta = csv.DictReader(metafile)
        for utt in meta:
            recoid = utt["file_name"]
            start = float(utt["start_time"])
            end = float(utt["end_time"])
            uttid = "{}_{:06d}_{:06d}".format(recoid, int(start * 100), int(end * 100))

            transcription = utt["tgt_lang"]
            transcription = transcription.replace("U", "_")
            transcription = transcription.lower()
            transcription = transcription.replace("_", "U")
            transcription = re.sub(punct_re, " ", transcription)
            transcription = " ".join(transcription.split())

            text.write("{} {}\n".format(uttid, transcription))
            utt2spk.write("{} {}\n".format(uttid, uttid))
            wavscp.write(
                '{} sox --norm=-1  "{}" -r 16k -t wav -c 1 -b 16 -e signed - trim {} {} |\n'.format(
                    uttid, wavs[recoid], start, end - start
                )
            )

    subprocess.call("utils/fix_data_dir.sh {}".format(odir), shell=True)
