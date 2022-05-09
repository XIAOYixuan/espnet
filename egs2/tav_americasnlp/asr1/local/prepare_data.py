#!/usr/bin/env python3

# Copyright 2022 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import csv
import glob
import os
import subprocess
import sys

idir = sys.argv[1]

sentences_nodevtest = set()
speakers_nodevtest = set()

with open("local/sentences_nodevtest.txt", encoding="utf-8") as f:
    for line in f:
        sentences_nodevtest.add(line.strip())

utts = []

with open(f"{idir}/pandialectal-bribri.tsv", encoding="utf-8") as metafile:
    meta = csv.DictReader(metafile, delimiter="\t")
    for row in meta:
        utts.append(row)
        if row["bribri_utf8"] in sentences_nodevtest:
            speakers_nodevtest.add(row["speaker_id"])

subsets = {"train": [], "dev": [], "test": []}

for s in [("test", 1000), ("dev", 250)]:
    while True:
        row = utts.pop()

        if row["bribri_utf8"] in sentences_nodevtest:
            subsets["train"].append(row)
        else:
            subsets[s[0]].append(row)

        if len(utts) == 0 or len(subsets[s[0]]) == s[1]:
            break
    if len(utts) == 0:
        break

for i in range(len(utts)):
    subsets["train"].append(utts[i])

for subset in ["train", "dev", "test"]:
    odir = "data/" + subset
    os.makedirs(odir, exist_ok=True)
    with open(odir + "/text", "w", encoding="utf-8") as text, open(
        odir + "/wav.scp", "w"
    ) as wavscp, open(odir + "/utt2spk", "w") as utt2spk:
        for utt in subsets[subset]:
            uttid = utt["speaker_id"] + "_" + utt["file"][:-4]
            text.write("{} {}\n".format(uttid, utt["bribri_utf8"]))
            utt2spk.write("{} {}\n".format(uttid, utt["speaker_id"]))
            wavscp.write("{} {}/{}\n".format(uttid, idir, utt["file"]))

    subprocess.call("utils/fix_data_dir.sh {}".format(odir), shell=True)
