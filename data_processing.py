#1forall 处理
import json
import argparse

def processing(args):
    with open("dataset.jsonl", 'r') as r:
        lines = r.readlines()
        b = int(len(lines) * 0.95)
        test = lines[b:]
    with open(args.pg_output, "r")as r:
        plots = r.readlines()

    for i in range(len(plots)):
        source = dict()
        plot = json.loads(plots[i])
        ori_story = json.loads(test[i])
        text = plot["pred"]
        events_list = text.split("[sep]")
        source["s0"] = ori_story["s0"]
        source["e1"] = events_list[0].replace("</s><s>", "").strip(" ")
        source["e2"] = events_list[1].strip(" ")
        source["e3"] = events_list[2].replace("</s>", "").replace("<pad>", "").strip(" ")
        try:
            source["e4"] = events_list[3].replace("</s>", "").replace("<pad>", "").strip(" ")
        except:
            pass
        with open(args.sr_input, "a")as w:
            w.write(json.dumps(source))
            w.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pg_output", type=str, required=True, help="path to pg output file")
    parser.add_argument("--sr_input", type=str, required=True, help="path to save file for sr test")

    args = parser.parse_args()

    processing(args)