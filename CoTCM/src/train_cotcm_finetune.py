# 导入必要的库和模块
import torch
from torch.nn.functional import softmax
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import Adafactor, AdamW
import sys, os
import pickle
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, Dataset)
from torch.utils.data.distributed import DistributedSampler
import joblib
import numpy as np
import argparse
import random
from torch.nn.parallel import DataParallel
from tqdm import tqdm, trange
import torch.nn as nn
import pdb
import pandas as pd
import json
from models.cotcm_moe_mor_t5 import HFMoET5Model

import torch
torch.cuda.empty_cache()
# 自定义 JSONL 数据集类
class JSONLDataset(Dataset):
    def __init__(self, pth, tokenizer, max_length=512, few=1.0):
        self.samples = []
        with open(pth, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append((item['input'], item['target']))
        self.tokenizer = tokenizer
        self.len = max_length
        self.few = few

    def __len__(self):
        return int(len(self.samples) * self.few)

    def __getitem__(self, index):
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len) * (-100)

        inp_text, out_text = self.samples[index]
        inp = self.tokenizer.encode(inp_text, truncation=True, max_length=self.len)
        out = self.tokenizer.encode(out_text, truncation=True, max_length=self.len)

        inp_ids[:len(inp)] = torch.tensor(inp)
        att_msk[:len(inp)] = 1
        labels[:len(out)] = torch.tensor(out)

        return inp_ids.long(), att_msk.long(), labels.long()

def clean_text(text):
    return text.replace('<pad>', '').replace('</s>', '').strip()

def safe_decode(tokenizer, ids):
    """安全解码，过滤非法 token ID，防止 out-of-range 报错"""
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    # 只保留在词表范围内的 token
    safe_ids = [i for i in ids if 0 <= i < tokenizer.vocab_size]
    return clean_text(tokenizer.decode(safe_ids, skip_special_tokens=True))

def do_eval(model, dataloader, tokenizer, pth, tag=0, iftest=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fw = open(pth, 'w', encoding='utf-8')
    fw.write('input\tground truth\toutput\n')
    model.eval()
    hitcnt = 1
    with torch.no_grad():
        for idx, i in enumerate(tqdm(dataloader)):
            inputs, atts, labels = i
            inputs, atts, labels = inputs.to(device), atts.to(device), labels.to(device)
            lab = (labels > 0).long() * labels
            output = model.generate(
                input_ids=inputs,
                attention_mask=atts,
                max_new_tokens=160,
                num_beams=1,  # 不使用 beam search
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.8,
                early_stopping=True,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                decoder_start_token_id=model.config.decoder_start_token_id,
                num_return_sequences=1  # 一条完整输出
            )

            # 解码 input、label、output
            for jdx, out in enumerate(output):
                fw.write(
                    safe_decode(tokenizer, inputs[jdx]) + '\t' +
                    safe_decode(tokenizer, lab[jdx]) + '\t' +
                    safe_decode(tokenizer, out) + '\n'
                )
    fw.close()


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda')

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    config = T5Config.from_pretrained("t5-small")

    config.num_experts = 6
    config.use_cache = True
    model = HFMoET5Model(config)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {args.init_checkpoint}")
        print(f"Missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}")

    model.to(device)


    global_step = 0



    # 检查参数比例
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())


    # 使用更小的学习率和梯度裁剪
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-5,  # 更小的学习率
        weight_decay=0.01
    )

    print('dataset loading')
    TrainSet = JSONLDataset(args.pth_train, tokenizer, few=args.few)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=2, pin_memory=True)
    DevSet = JSONLDataset(args.pth_dev, tokenizer)
    dev_dataloader = DataLoader(DevSet, shuffle=False,
                                batch_size=args.batch_size, drop_last=False,
                                num_workers=2, pin_memory=True)
    TestSet = JSONLDataset(args.pth_test, tokenizer)
    test_dataloader = DataLoader(TestSet, shuffle=False,
                                 batch_size=args.batch_size, drop_last=False,
                                 num_workers=2, pin_memory=True)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    avg_loss = 0
    tag = 0
    max_rec = 0
    early = 0
    print('start training')
    accu = 16 / args.batch_size
    for epoch in range(args.epoch):
        model.train()
        if tag:
            break
        print('epoch: ', epoch)
        for idx, i in enumerate(tqdm(train_dataloader)):
            if tag:
                break
            inputs, atts, labels = i
            inputs = inputs.to(device)
            atts = atts.to(device)
            labels = labels.to(device)

            output = model(input_ids=inputs, attention_mask=atts, labels=labels, return_dict=True)
            loss = output.loss.mean() / accu
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            avg_loss += loss.item()
            if idx % accu == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % 25 == 0:
                    print('Step: ', global_step, ', loss: ', avg_loss)
                    avg_loss = 0
                if global_step > args.global_step:
                    return
                if global_step % 1000 == 0:
                    # 创建保存模型文件的路径，包含 global_step 或者 epoch 信息
                    save_path = os.path.join(save_dir, f"model_step_{global_step}.pt")

                    # 保存模型参数
                    torch.save(model.state_dict(), save_path)  # 推荐保存模型的参数而不是整个模型对象
                    print(f"Model saved at step {global_step} to {save_path}")
                    #evs = do_eval(model, dev_dataloader, tokenizer, args.pth_out)
                    #model.train()
                    #print('Step:', global_step, evs, early)

        torch.cuda.empty_cache()
    # = do_eval(model, dev_dataloader, tokenizer, args.pth_out)
    #print('Step:', global_step, evs)

    #torch.save(model, args.save_pth)
    #model.load_state_dict(torch.load(args.save_pth, weights_only=False).state_dict())
    #evs = do_eval(model, test_dataloader, tokenizer, args.pth_out, iftest=True)
    #print('Test: ', evs)


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--save_pth", default='save_model/final_moe_MoR2.pt', type=str)
    parser.add_argument("--init_checkpoint", default=None, type=str)
    parser.add_argument("--save_dir", default='checkpoints', type=str)

    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--pth_train", default='data/fine_tuning/train_data_separated.jsonl', type=str)
    parser.add_argument("--pth_dev", default='data/fine_tuning/dev_data_separated.jsonl', type=str)
    parser.add_argument("--pth_test", default='data/fine_tuning/test_data_separated.jsonl', type=str)
    parser.add_argument("--pth_out", default='log/moe_output.txt', type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--epoch", default=100, type=int)
    parser.add_argument("--seed", default=1111, type=int)
    parser.add_argument("--global_step", default=60000, type=int)
    parser.add_argument("--few", default=1.0, type=float)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
