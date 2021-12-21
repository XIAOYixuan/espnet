#!/usr/bin/env python3

# Copyright 2021 University of Stuttgart (Pavel Denisov)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser(
    description="Prepare Switchboard Dialogue Act dataset."
)

parser.add_argument("audio_path", type=str, help="Path to audio (LDC97S62)")
parser.add_argument("nxt_path", type=str, help="Path to NXT annotation (LDC2009T26)")
parser.add_argument(
    "--context", type=int, default=1, help="Number of utterances in the context"
)
parser.add_argument(
    "--audio-context", type=float, default=0.0, help="Length of audio in the context"
)

args = parser.parse_args()

if args.context > 1 and args.audio_context > 0.0:
    sys.exit("Do not set both context and audio context at the same time")

xml_path = os.path.join(args.nxt_path, "nxt_switchboard_ann", "xml")

channel = {"A": 1, "B": 2}
speaker = {}

corpus_resources_root = ET.parse(
    os.path.join(xml_path, "corpus-resources", "dialogues.xml")
).getroot()
for dialogue in corpus_resources_root.findall(".//dialogue"):
    dialogue_id = "sw" + dialogue.attrib["swbdid"]
    speaker[dialogue_id] = {}
    for pointer in dialogue.findall(".//{http://nite.sourceforge.net/}pointer"):
        speaker[dialogue_id][pointer.attrib["role"]] = pointer.attrib["href"].split(
            "#"
        )[1][3:-1]

sph = {}

for sph_file in glob.glob(os.path.join(args.audio_path, "*/swb1/sw*.sph")):
    dialogue_id = sph_file[-10:-4]
    sph[dialogue_id] = sph_file

# Data splits local/{train,valid,test}.lst
# from the paper: Ji Young Lee*, Franck Dernoncourt*.
# Sequential Short-Text Classification with Recurrent and Convolutional Neural Networks.
# NAACL 2016. (* indicates equal contribution)

for subset in ["train", "valid", "test"]:
    subset_dir = subset

    if args.context > 1:
        subset_dir += "_context" + str(args.context)
    elif args.audio_context > 0.0:
        subset_dir += "_audio{:03}".format(int(args.audio_context * 10))

    subset_dir = os.path.join("data", subset_dir)
    os.makedirs(subset_dir, exist_ok=True)

    with open(os.path.join(subset_dir, "text"), "w") as text_f, open(
        os.path.join(subset_dir, "wav.scp"), "w"
    ) as wav_scp_f, open(
        os.path.join(subset_dir, "utt2spk"), "w"
    ) as utt2spk_f, open(
        os.path.join("local", subset + ".lst")
    ) as dialogues_f:
        for line in dialogues_f:
            dialogue_id = line.strip()

            dial_acts = {}

            for role in ["A", "B"]:
                terminals = {}

                terminals_file = os.path.join(
                    xml_path, "terminals", f"{dialogue_id}.{role}.terminals.xml"
                )

                if not os.path.exists(terminals_file):
                    continue

                terminals_root = ET.parse(terminals_file).getroot()

                for terminal in terminals_root.findall(".//word"):
                    start_str = terminal.attrib["{http://nite.sourceforge.net/}start"]
                    end_str = terminal.attrib["{http://nite.sourceforge.net/}end"]
                    if (
                        start_str != "non-aligned"
                        and start_str != "n/a"
                        and end_str != "n/a"
                    ):
                        terminals[
                            terminal.attrib["{http://nite.sourceforge.net/}id"]
                        ] = {
                            "start": float(start_str),
                            "end": float(end_str),
                        }

                dial_act_root = ET.parse(
                    os.path.join(
                        xml_path, "dialAct", f"{dialogue_id}.{role}.dialAct.xml"
                    )
                ).getroot()

                for dial_act in dial_act_root.findall(".//da"):
                    words = dial_act.attrib["niteType"]
                    if words == "excluded":
                        continue

                    dial_act_id = dial_act.attrib["{http://nite.sourceforge.net/}id"][
                        2:
                    ]

                    utt_id = (
                        speaker[dialogue_id][role]
                        + "_"
                        + dialogue_id
                        + "_"
                        + dial_act_id
                    )

                    dial_act_children = dial_act.findall(
                        ".//{http://nite.sourceforge.net/}child"
                    )

                    start_terminal_id = (
                        dial_act_children[0].attrib["href"].split("#")[1][3:-1]
                    )
                    end_terminal_id = (
                        dial_act_children[-1].attrib["href"].split("#")[1][3:-1]
                    )

                    if (
                        start_terminal_id not in terminals
                        or end_terminal_id not in terminals
                    ):
                        continue

                    start = terminals[start_terminal_id]["start"]
                    end = terminals[end_terminal_id]["end"]
                    dur = end - start

                    if dur < 0.005:
                        continue

                    dial_acts[dial_act_id] = {
                        "utt": utt_id,
                        "start": start,
                        "end": end,
                        "dur": dur,
                        "channel": channel[role],
                        "text": words,
                        "spk": speaker[dialogue_id][role],
                    }

            for dial_act_id in dial_acts:
                if args.audio_context > 0.0:
                    start = max(dial_acts[dial_act_id]["end"] - args.audio_context, 0.0)
                    dur = dial_acts[dial_act_id]["end"] - start
                    wav_scp_f.write(
                        "{} sox {} -r 16k -t wav -c 1 -b 16 -e signed - trim {} {} |\n".format(
                            dial_acts[dial_act_id]["utt"], sph[dialogue_id], start, dur)
                    )
                else:
                    context = [dial_act_id]

                    for i in range(1, args.context):
                        context_dial_act_id = str(int(dial_act_id) - i)
                        if context_dial_act_id in dial_acts:
                            context.append(context_dial_act_id)

                    context.reverse()

                    wav = " ".join(
                        [
                            '"| sox {} -r 16k -t wav -c 1 -b 16 -e signed - '
                            + 'trim {} {} remix {}"'.format(
                                sph[dialogue_id],
                                dial_acts[c]["start"],
                                dial_acts[c]["dur"],
                                dial_acts[c]["channel"],
                            )
                            for c in context
                        ]
                    )

                    wav_scp_f.write(
                        "{} sox {} -t wav - |\n".format(dial_acts[dial_act_id]["utt"], wav)
                    )

                text_f.write(
                    dial_acts[dial_act_id]["utt"]
                    + " "
                    + dial_acts[dial_act_id]["text"]
                    + "\n"
                )
                utt2spk_f.write(
                    dial_acts[dial_act_id]["utt"]
                    + " "
                    + dial_acts[dial_act_id]["spk"]
                    + "\n"
                )
