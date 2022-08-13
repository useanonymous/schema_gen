import torch
import argparse
import code
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset


def collate_fn(batch):
    sorted_by_input= sorted(batch, key=lambda x: len(x[0]), reverse=True)
    input, label = zip(*batch)
    pad_input = []
    pad_label = []
    max_input_len = len(sorted_by_input[0][0])

    for i in range(len(input)):
        temp_input = [-100] * max_input_len
        temp_input[:len(input[i])] = input[i]
        pad_input.append(temp_input)

        temp_label = [-100] * max_input_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)

    
    return pad_input, pad_label

class GPT2Dataset_all(Dataset):

    def __init__(self, txt_list, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels_ids = []
        for j in txt_list:
            row = json.loads(j)
            source = f"[e] {row['e1']} [e] {row['e2']} [e] {row['e3']} [e] {row['e4']} [sep] {row['s0']}"
            target = f"{row['s1']} {row['s2']} {row['s3']} {row['s4']}"
            
            source_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source))
            target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))

            label = [tokenizer.pad_token_id]*len(source_ids) + target_ids + [tokenizer.eos_token_id]
            encoder_input = source_ids + [tokenizer.bos_token_id] + target_ids

            if len(label)> 512:
                label = label[-512:]
                encoder_input = encoder_input[-512:]
            
            self.input_ids.append(encoder_input)
            self.labels_ids.append(label)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels_ids[idx]

def test(args):
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(args.checkpoint_path)
    model = GPT2LMHeadModel.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    model.eval()
    
    
    with open(args.input_path, 'r') as r:
        lines = r.readlines()
    for i in lines:
        row = json.loads(i)
        if row.__contains__("e4"):
            input = f"[e] {row['e1']} [e] {row['e2']} [e] {row['e3']} [e] {row['e4']} [sep] {row['s0']}"
        else:
            input = f"[e] {row['e1']} [e] {row['e2']} [e] {row['e3']} [sep] {row['s0']}"
        input_dict = tokenizer(input, return_tensors="pt")
        input_ids = input_dict["input_ids"].to(device)
        attention_mask = input_dict["attention_mask"].to(device)
        outputs  = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=512,
            do_sample=True,
            top_p=args.top_p,
            top_k=args.top_k,
            temperature=args.temperature
        )
        preds = [tokenizer.decode(ids, skip_special_tokens=False) for ids in outputs]
        res = dict()
        for m in preds:
            res = dict()
            res['pred'] = m
            with open(args.output_path, "a")as w:
                w.write(json.dumps(res, ensure_ascii=False))
                w.write('\n')
        torch.cuda.empty_cache()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to saved checkpoint file")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=float, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_path", type=str, default='output.jsonl')
    parser.add_argument("--input_path", type=str, required=True)

    args = parser.parse_args()

    test(args)