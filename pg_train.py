import random
import torch
import argparse
import os
import time
import datetime
import code
import json
import torch.cuda.amp as amp
import torch.nn.functional as F
import numpy as np
from transformers import BartTokenizer, BartForConditionalGeneration, AdamW
from torch.utils.data import DataLoader, SequentialSampler, Dataset, RandomSampler

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

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

        temp_decoder_attn= [0] * max_label_len
        temp_decoder_attn[:len(decoder_input[i])] = decoder_attn[i]
        pad_decoder_attn.append(temp_decoder_attn)

    
    return pad_input, pad_label, pad_decoder_input, pad_input_attn, pad_decoder_attn

class Get_Dataset(Dataset):

    def __init__(self, args, txt_list, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.decoder_input_ids = []
        self.labels_ids = []
        self.input_attn = []
        self.decoder_attn = []
        for row in txt_list:
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


def forward_step(model, tokenizer, batch_data):
    input_ids = torch.LongTensor(batch_data[0])
    labels = torch.LongTensor(batch_data[1])
    decoder_input_ids = torch.LongTensor(batch_data[2])
    input_attention_mask = torch.LongTensor(batch_data[3])
    decoder_attention_mask = torch.LongTensor(batch_data[4])

    input_ids[input_ids[:] == -100] = tokenizer.pad_token_id
    labels[labels[:] == -100] = tokenizer.pad_token_id
    decoder_input_ids[decoder_input_ids[:] == -100] = tokenizer.pad_token_id

    device=model.device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    decoder_input_ids = decoder_input_ids.to(device)
    input_attention_mask = input_attention_mask.to(device)
    decoder_attention_mask = decoder_attention_mask.to(device)

    outputs = model(
        input_ids = input_ids,
        attention_mask = input_attention_mask,
        decoder_input_ids = decoder_input_ids,
        decoder_attention_mask = decoder_attention_mask,
        )

    loss = F.cross_entropy(
        outputs["logits"].view(-1, outputs["logits"].size(-1)),
        labels.view(-1),
        ignore_index= tokenizer.pad_token_id,
        reduction="mean"
        )
    with torch.no_grad():
        ppl = loss.exp()

    return loss, ppl

def train(args):

    # set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda')

    data =[]
    with open(args.datapath, 'r') as r:
        lines = r.readlines()
    for line in lines:
        data.append(json.loads(line))
    a, b = int(len(data) * 0.9), int(len(data) * 0.95)
    train, val, test = data[:a], data[a:b], data[b:]

    #load pre-trained model
    if args.checkpoint_path:
        tokenizer = BartTokenizer.from_pretrained(args.checkpoint_path)
        model = BartForConditionalGeneration.from_pretrained(args.checkpoint_path)
    else:
        tokenizer = BartTokenizer.from_pretrained(args.pretrain_model, sep_token="[sep]")
        tokenizer.add_tokens(['[k]', '[e]'])
        model = BartForConditionalGeneration.from_pretrained(args.pretrain_model)
        model.resize_token_embeddings(len(tokenizer))

    model = model.to(device)

    #load train and val data
    train_dataset = Get_Dataset(args=args, txt_list=train, tokenizer=tokenizer)
    val_dataset = Get_Dataset(args=args, txt_list=val, tokenizer=tokenizer)

    #training data batch
    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        collate_fn=collate_fn,
        batch_size = args.batch_size
    )

    validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            collate_fn=collate_fn,
            batch_size = args.batch_size
        )


    optimizer = AdamW(model.parameters(),
                  lr = args.learning_rate
                )
    scaler = amp.GradScaler()


    for k, v in sorted(dict(args.__dict__).items()):
        print("{}: {}".format(k, v))
    
    #start time for whole training
    t0 = time.time()

    steps = 0
    for _ in range(args.epochs):
        # ========================================
        #               Training
        # ========================================

        for batch in train_dataloader:
            steps += 1
            model.train()
            
            with amp.autocast():
                loss, ppl = forward_step(model, tokenizer, batch)

            batch_loss = loss.item()
            batch_ppl = ppl.item()

            #print state
            if steps % args.print_every == 0:
                batch_loss = loss.item()
                batch_ppl = ppl.item()
                #train time for one epoch
                elapsed = format_time(time.time() - t0)
                print('  steps {:>5,}.  Loss: {:>5,}. PPL: {:>5,}. Elapsed: {:}  '.format(steps, batch_loss, batch_ppl, elapsed))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # ========================================
            #               Validation
            # ========================================
            if steps % args.eval_every == 0:  

                model.eval()
                total_eval_loss = 0
                total_eval_ppl = 0
                #evaluate data for one epoch
                for eval_step, batch in enumerate(validation_dataloader):
                    with amp.autocast():
                        with torch.no_grad():
                            loss, ppl =forward_step(model, tokenizer, batch)
                            batch_loss = loss.item()
                            total_eval_loss += batch_loss
                            batch_ppl = ppl.item()
                            total_eval_ppl += batch_ppl

                # Calculate the average loss over all of the batches in val.
                avg_val_loss = total_eval_loss / eval_step
                avg_val_ppl = total_eval_ppl / eval_step
    
                validation_time = format_time(time.time() - t0)    

                print('  VAl Loss: {:>5,}. PPL: {:>5,}. Elapsed: {:}  '.format(avg_val_loss, avg_val_ppl, validation_time))

                #model save path
                if args.save_model:
                    output_dir = f"{args.save_path}/model/"
                    #Create output directory if needed
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    # Save a trained model
                    model.save_pretrained(f"{output_dir}")
                    tokenizer.save_pretrained(f"{output_dir}")
                    print("model saved")
            torch.cuda.empty_cache()

    print("")
    print("Training complete!")
    print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-t0)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Some hyperparas=meters")
    parser.add_argument('--seed', type=int, default=42,
                    help='pytorch seed')
    parser.add_argument('--pretrain_model', type=str, required=True,
                    help='pytorch seed')
    parser.add_argument('--datapath', type=str, default="dataset.jsonl",
                    help='path of training dataset')
    parser.add_argument('--checkpoint_path', type=str,
                    help='path of echeckpoint')                    
    parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size')
    parser.add_argument('--save_path', type=str, default="save",
                    help='path of saving model params')                 
    parser.add_argument('--epochs', type=int, default=5,
                    help='epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate of optimizer')
    parser.add_argument('--print_every', type=int, default=500,
                    help='print evert steps')
    parser.add_argument('--eval_every', type=int, default=1000,
                    help='evaluate every steps')
    parser.add_argument("--save_model", action="store_true", default=False)
    parser.add_argument("--num_of_knows", type=int, default=60)
    args = parser.parse_args()

    train(args)