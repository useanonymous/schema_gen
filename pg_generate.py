import json
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import Dataset, SequentialSampler


class Get_Dataset(Dataset):

    def __init__(self, args, txt_list, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.decoder_input_ids = []
        self.labels_ids = []
        self.input_attn = []
        self.decoder_attn = []
        for j in txt_list:
            row = json.loads(j)
            schema = " [k] ".join(row['schema'][:args.num_of_knows])
            source = f"[k] {schema} [e] {row['e0']}"

            target = f"{row['e1']} [sep] {row['e2']} [sep] {row['e3']} [sep] {row['e4']}"
            
            source_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source))
            target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))
            
            self.input_ids.append(source_ids)
            self.labels_ids.append(target_ids + [tokenizer.eos_token_id])
            self.decoder_input_ids.append([tokenizer.bos_token_id]+target_ids)
            self.input_attn.append([1 for _ in range(len(source_ids))])
            self.decoder_attn.append([1 for _ in range(len(target_ids)+1)])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels_ids[idx], self.decoder_input_ids[idx], self.input_attn[idx], self.decoder_attn[idx]


def data_load(args, tokenizer, model):
    data = []
    with open(args.data_dir, 'r') as r:
        lines = r.readlines()
    a, b = int(len(lines) * 0.9), int(len(lines) * 0.95)
    test_ori = lines[b:] # 0.9, 0.05, 0.05
    test = Get_Dataset(args=args, txt_list=test_ori, tokenizer=tokenizer)

    return test


def collate_fn(batch):
    sorted_by_label = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    sorted_by_input = sorted(batch, key=lambda x: len(x[0]), reverse=True)
    input, label, decoder_input, input_attn, decoder_attn = zip(*batch)
    pad_input = []
    pad_label = []
    pad_decoder_input = []
    pad_input_attn = []
    pad_decoder_attn = []
    max_input_len = len(sorted_by_input[0][0])
    max_label_len = len(sorted_by_label[0][1])
    for i in range(len(input)):
        temp_input = [-100] * max_input_len
        temp_input[:len(input[i])] = input[i]
        pad_input.append(temp_input)

        temp_label = [-100] * max_label_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)

        temp_decoder_input = [-100] * max_label_len
        temp_decoder_input[:len(decoder_input[i])] = decoder_input[i]
        pad_decoder_input.append(temp_decoder_input)

        temp_input_attn = [0] * max_input_len
        temp_input_attn[:len(input_attn[i])] = input_attn[i]
        pad_input_attn.append(temp_input_attn)

        temp_decoder_attn = [0] * max_label_len
        temp_decoder_attn[:len(decoder_attn[i])] = decoder_attn[i]
        pad_decoder_attn.append(temp_decoder_attn)
    
    return pad_input, pad_label, pad_decoder_input, pad_input_attn, pad_decoder_attn

def test(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda')
    tokenizer = BartTokenizer.from_pretrained(args.checkpoint_path)
    model = BartForConditionalGeneration.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    model.eval()

    test_data = data_load(args, tokenizer, model)

    test_dataloader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        sampler = SequentialSampler(test_data),
        collate_fn=collate_fn,
        num_workers=0
    )
    if args.generate:
        for batch_data in tqdm(test_dataloader):
            input_ids = torch.LongTensor(batch_data[0]).to(device)
            input_ids[input_ids[:,:] == -100] = tokenizer.pad_token_id
            labels = torch.LongTensor(batch_data[1]).to(device)
            labels[labels[:] == -100] = tokenizer.pad_token_id
            decoder_input_ids = torch.LongTensor(batch_data[2]).to(device)
            decoder_input_ids[decoder_input_ids[:] == -100] = tokenizer.pad_token_id
            input_attn = torch.LongTensor(batch_data[3]).to(device)
            decoder_attn = torch.LongTensor(batch_data[4]).to(device)

            outputs  = model.generate(input_ids,
                                    attention_mask=input_attn,
                                    max_length=args.max_gen_length,
                                    do_sample=True,
                                    top_k=args.top_k,
                                    top_p=args.top_p,
                                    temperature=args.temperature)
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

    parser.add_argument("--seed", type=int, default=42, help="random initialization seed")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="path to saved checkpoint file")
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_gen_length", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default='dataset.jsonl')
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for generation")
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--output_path", type=str, default='output.jsonl')
    parser.add_argument("--num_of_knows", type=int, default=60)
    parser.add_argument("--generate", action="store_true", default=False)

    args = parser.parse_args()

    test(args)