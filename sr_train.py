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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset


def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

def collate_fn(batch):
    sorted_by_input= sorted(batch, key=lambda x: len(x[0]), reverse=True)
    input, label, attn = zip(*batch)
    pad_input = []
    pad_label = []
    pad_attn = []
    max_input_len = len(sorted_by_input[0][0])

    for i in range(len(input)):
        temp_input = [-100] * max_input_len
        temp_input[:len(input[i])] = input[i]
        pad_input.append(temp_input)

        temp_label = [-100] * max_input_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)

        temp_attn = [0] * max_input_len
        temp_attn[:len(attn[i])] = attn[i]
        pad_attn.append(temp_attn)

    
    return pad_input, pad_label, pad_attn

class GPT2Dataset(Dataset):

    def __init__(self, txt_list, tokenizer):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.labels_ids = []
        self.attention_mask = []
        for j in txt_list:
            row = json.loads(j)
            source = f"{row['s0']}"
            target = f"{row['s1']} {row['s2']} {row['s3']} {row['s4']}"
            
            source_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(source))
            target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(target))

            label = [tokenizer.pad_token_id]*len(source_ids) + target_ids + [tokenizer.eos_token_id]
            input = source_ids + [tokenizer.bos_token_id] + target_ids
            attention = [1 for _ in range(len(input))]

            if len(label)> 512:
                label = label[-512:]
                input = input[-512:]
                attention =attention[-512:]
            
            self.input_ids.append(input)
            self.labels_ids.append(label)
            self.attention_mask.append(attention)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels_ids[idx], self.attention_mask[idx]

def forward_step(model, tokenizer, batch_data):
    input_ids = torch.LongTensor(batch_data[0])
    labels = torch.LongTensor(batch_data[1])
    attention_mask = torch.LongTensor(batch_data[2])

    input_ids[input_ids[:,:] == -100] = tokenizer.pad_token_id
    labels[labels[:,:] == -100] = tokenizer.pad_token_id

    device=model.device
    input_ids = input_ids.to(device)
    labels = labels.to(device)
    attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids = input_ids,
        attention_mask = attention_mask,
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

    device = torch.device("cuda")

    data =[]
    with open(args.datapath, 'r') as r:
        lines = r.readlines()
    for line in lines:
        data.append(line)
    a, b = int(len(data) * 0.9), int(len(data) * 0.95)
    train, val, test = data[:a], data[a:b], data[b:]

    #load pre-trained model
    tokenizer = GPT2Tokenizer.from_pretrained(args.pretrain_model, sep_token="[sep]", pad_token="[pad]", unk_token="[unk]", bos_token="[bos]")
    #tokenizer.add_tokens(['[e]'])

    model = GPT2LMHeadModel.from_pretrained(args.pretrain_model)
    model.resize_token_embeddings(len(tokenizer))
    #model = model.half()
    model = model.to(device)

    #load train and val data
    train_dataset = GPT2Dataset(txt_list=train, tokenizer=tokenizer)
    val_dataset = GPT2Dataset(txt_list=val, tokenizer=tokenizer)

    #training data batch
    train_dataloader = DataLoader(
        train_dataset,
        sampler = RandomSampler(train_dataset),
        collate_fn=collate_fn,
        batch_size = args.batch_size
    )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    validation_dataloader = DataLoader(
            val_dataset, 
            sampler = SequentialSampler(val_dataset), 
            collate_fn=collate_fn,
            batch_size = args.batch_size
        )


    optimizer = torch.optim.AdamW(model.parameters(),
                  lr = args.learning_rate
                )
    scaler = amp.GradScaler()
    
    #start time for whole training
    t0 = time.time()

    #tensorboard log path
    for k, v in sorted(dict(args.__dict__).items()):
        print("{}: {}".format(k, v))

    steps = 0
    for _ in range(args.epochs):
        
        # ========================================
        #               Training
        # ========================================

        #start time of one epoch in train

        for batch in train_dataloader:
            steps += 1
            model.train()
            optimizer.zero_grad()
            
            with amp.autocast():
                loss, ppl = forward_step(model, tokenizer, batch)

            batch_loss = loss.item()
            batch_ppl = ppl.item()

            #print state
            if steps % args.print_every == 0:
                #train time for one epoch
                elapsed = format_time(time.time() - t0)
                print('  steps {:>5,}.  Loss: {:>5,}. PPL: {:>5,}. Elapsed: {:}  '.format(steps, batch_loss, batch_ppl, elapsed))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if steps % args.eval_every == 0:  
            # ========================================
            #               Validation
            # ========================================

                #start time of one epoch in validation

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
                # Measure how long val took in this epoch.        
                validation_time = format_time(time.time() - t0)    

                print('  VAl Loss: {:>5,}. PPL: {:>5,}. Elapsed: {:}  '.format(avg_val_loss, avg_val_ppl, validation_time))
                
                #model save path
                if args.save_model:
                    output_dir = f"{args.save_path}/model/"
                    #Create output directory if needed
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    print("Saving model to %s" % output_dir)

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
    parser.add_argument('--pretrain_model', type=str, required=True)
    parser.add_argument('--datapath', type=str, default="dataset.jsonl",
                    help='path of training dataset')
    parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
    parser.add_argument('--save_path', type=str, default="save",
                    help='path of saving model params')                 
    parser.add_argument('--epochs', type=int, default=5,
                    help='epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                    help='learning rate of optimizer')
    parser.add_argument('--print_every', type=int, default=100,
                    help='print evert steps')
    parser.add_argument('--eval_every', type=int, default=200,
                    help='evaluate every steps')
    parser.add_argument("--save_model", action="store_true", default=False)     
    args = parser.parse_args()

    train(args)