#!/usr/bin/env python3

from multiprocessing import Pool
import os
import queue
import sys

from Preprocessing.ArticulatoryCombinedTextFrontend import (
    ArticulatoryCombinedTextFrontend,
)

itext = sys.argv[1]
otext = sys.argv[2]

ft = ArticulatoryCombinedTextFrontend(language="en")

def phonemize_utt(line):
    utt, text = line.strip().split(" ", maxsplit=1)
    text_phn = ft.get_phone_string(text).rstrip("#").strip("~")
    line_phn = "{} {}\n".format(utt, text_phn)
    return line_phn

lines = []

with open(itext, encoding="utf=8") as f:
    for line in f:
        lines.append(line.strip())

with Pool(16) as p:
    output = p.map(phonemize_utt, lines)

with open(otext, "w", encoding="utf=8") as f:
    f.writelines(output)
