from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler
import joblib
import random
import numpy as np
import math
from tqdm import tqdm
import argparse
from transformers import T5Tokenizer, T5Config
from transformers.models.t5.modeling_t5 import T5Stack, T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
import torch
import torch.nn as nn
from transformers import T5PreTrainedModel, T5Config
from transformers.models.t5.modeling_t5 import T5LayerNorm, T5LayerFF, T5Attention
import copy
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
import torch.nn.functional as F
import torch
torch.cuda.empty_cache()  # 释放未使用的缓存
from transformers.modeling_utils import PreTrainedModel


class TransformerExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm1 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.ff = T5LayerFF(config)
        self.layer_norm2 = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states, attention_mask=None):
        norm_x = self.layer_norm1(hidden_states)
        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[:, None, None, :]
        attn_output = self.self_attn(
            norm_x,
            mask=attention_mask,
            position_bias=None,
            past_key_value=None,
            layer_head_mask=None,
            output_attentions=False,
            use_cache=False
        )[0]
        hidden_states = hidden_states + self.dropout(attn_output)
        norm_x = self.layer_norm2(hidden_states)
        ff_output = self.ff(norm_x)
        hidden_states = hidden_states + self.dropout(ff_output)
        return hidden_states


class GatingNetwork(nn.Module):
    def __init__(self, config, num_experts, top_k=2):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.d_ff,
                dropout=config.dropout_rate,
                activation="gelu",
                batch_first=True
            ),
            num_layers=2
        )
        self.classifier = nn.Linear(config.d_model, num_experts)
        self.top_k = top_k
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        self.last_balancing_loss = None  # 👈 保存负载均衡 loss

    def forward(self, hidden_states, attention_mask=None):
        x = hidden_states.transpose(0, 1)  # [L, B, D]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [B, L, D]
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=-1)

        if self.top_k < probs.shape[-1]:
            topk = torch.topk(probs, self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topk.indices, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)

        self.last_gate_weights = probs.detach()
        return probs


class MoEBlock(nn.Module):
    def __init__(self, config, experts, gating, top_k=2):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.experts = experts
        self.gating = gating
        self.top_k = top_k

    def forward(self, hidden_states, attention_mask=None):
        normed = self.layer_norm(hidden_states)
        gate_logits = self.gating(normed, attention_mask)  # [B, L, E]
        topk_vals, topk_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_vals, dim=-1)  # [B, L, k]

        B, L, D = normed.shape
        moe_output = torch.zeros_like(normed)

        for k in range(self.top_k):
            expert_ids = topk_indices[:, :, k]  # [B, L]
            expert_weight = topk_weights[:, :, k]  # [B, L]

            for i, expert in enumerate(self.experts):
                # 创建一个 mask: [B, L]，表示哪些位置选择了当前专家
                mask = (expert_ids == i)
                if mask.sum() == 0:
                    continue

                selected_tokens = normed[mask]  # [N, D]
                selected_mask = None
                if attention_mask is not None:
                    selected_mask = attention_mask[mask]  # [N]

                expert_out = expert(selected_tokens.unsqueeze(0), selected_mask.unsqueeze(0) if selected_mask is not None else None)[0]
                expert_out = expert_out.squeeze(0)

                # 加权输出写回
                weighted_out = expert_out * expert_weight[mask].unsqueeze(-1)  # [N, D]
                moe_output[mask] += weighted_out

        return hidden_states + self.dropout(moe_output)



class MoEEncoder(PreTrainedModel):
    main_input_name = "input_ids"

    def __init__(self, config, shared, experts, gating, num_layers=6, top_k=2):
        super().__init__(config)
        self.shared = shared
        self.layers = nn.ModuleList([
            MoEBlock(config, experts, gating, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
    ):
        if attention_mask is not None and input_ids.shape[0] != attention_mask.shape[0]:
            expand_factor = input_ids.shape[0] // attention_mask.shape[0]
            attention_mask = attention_mask.unsqueeze(1).expand(-1, expand_factor, -1).reshape(input_ids.shape[0], -1)

        inputs_embeds = self.shared(input_ids)
        hidden_states = inputs_embeds
        all_hidden_states = [] if output_hidden_states else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.final_layer_norm(hidden_states)

        if not return_dict:
            return (hidden_states, all_hidden_states) if output_hidden_states else (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=None  # 你没有 attentions 的实现，直接写 None 即可
        )


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    return shifted_input_ids


class HFMoET5Model(T5PreTrainedModel, GenerationMixin):
    def __init__(self, config, num_experts=6, num_layers=6, top_k=2):
        super().__init__(config)
        config.is_encoder_decoder = True
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        self.experts = nn.ModuleList([TransformerExpert(config) for _ in range(num_experts)])
        self.gating = GatingNetwork(config, num_experts, top_k=top_k)
        self.encoder = MoEEncoder(config, self.shared, self.experts, self.gating, num_layers=num_layers, top_k=top_k)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.use_cache = True
        decoder_config.is_encoder_decoder = False
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        labels=None,
        past_key_values=None,
        use_cache=True,
        **kwargs
    ):
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask)
        encoder_hidden_states = encoder_outputs.last_hidden_state

        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = shift_tokens_right(
                labels,
                pad_token_id=self.config.pad_token_id,
                decoder_start_token_id=self.config.decoder_start_token_id,
            )

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        sequence_output = decoder_outputs[0]
        logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values if use_cache else None,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=encoder_hidden_states,
            encoder_hidden_states=encoder_outputs.hidden_states if encoder_outputs else None,
            encoder_attentions=encoder_outputs.attentions if encoder_outputs else None,
        )

    def _reorder_cache(self, past_key_values, beam_idx):
        if past_key_values is None:
            return None
        reordered_past = []
        for layer_past in past_key_values:
            reordered_past.append(
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past)
            )
        return reordered_past

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, encoder_outputs=None, **kwargs):
        return {
            "input_ids": None,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "use_cache": True
        }


# 自定义数据集类保持不变
class pre_dataset(Dataset):
    def __init__(self, pth, pth2, pth3, max_length=512):
        self.data = joblib.load(pth)
        tmp = joblib.load(pth2)
        self.data[0] += tmp[0]
        self.data[1] += tmp[1]
        tmp = joblib.load(pth3)
        self.data[0] += tmp[0]
        self.data[1] += tmp[1]
        self.len = max_length

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        inp_ids = torch.ones(self.len)
        att_msk = torch.zeros(self.len)
        labels = torch.ones(self.len) * (-100)
        inp = self.data[0][index]
        out = self.data[1][index]
        inp_ids[:min(self.len, len(inp))] = torch.tensor(inp[:min(self.len, len(inp))])
        att_msk[:min(self.len, len(inp))] = 1
        labels[:min(self.len, len(out))] = torch.tensor(out[:min(self.len, len(out))])
        return inp_ids.long(), att_msk.long(), labels.long()

# 训练主函数
def main(args):
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('加载tokenizer...')
    # 初始化 tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    # 创建模型配置（确保和你的自定义配置兼容）
    config = T5Config.from_pretrained("t5-small")

    config.num_experts = 6  # 设置专家数量，与你模型中一致
    config.use_cache = True  # 允许使用 past_key_values

    # 初始化模型
    model = HFMoET5Model(config)

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)

    print('Dataset loading...')
    TrainSet = pre_dataset(args.pth, args.pth2, args.pth3)
    train_sampler = RandomSampler(TrainSet)
    train_dataloader = DataLoader(TrainSet, sampler=train_sampler, batch_size=2,
                                  drop_last=False, num_workers=4, pin_memory=True)

    avg_loss = 0
    global_step = 0
    accu = 256 / args.batch_size
    print('开始训练...')
    for epoch in range(args.epoch):
        print(f'第 {epoch} 轮训练')
        for idx, (inputs, atts, labels) in enumerate(tqdm(train_dataloader)):
            inputs, atts, labels = inputs.to(device), atts.to(device), labels.to(device)
            outputs = model(input_ids=inputs, attention_mask=atts, decoder_input_ids=None, labels=labels,return_dict=True)

            loss= outputs.loss

            if math.isnan(loss.item()):
                loss = None
                continue
            loss.backward()
            avg_loss += loss.item()

            # 梯度累积
            if (idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # 打印训练信息
                if global_step % 25 == 0:
                    print(f'步骤: {global_step}, 损失: {avg_loss / 25:.4f}')
                    avg_loss = 0

                # 保存模型
                if global_step % 300 == 0 and global_step > args.step_pre:
                    torch.save(model.state_dict(), args.save_pth)
                    print(f'模型已保存至 {args.save_pth}')


                # 检查是否完成训练
                if global_step >= args.global_step:
                    return

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", default=None, type=str)
    parser.add_argument("--save_pth", default='save_model/moe_t5_small_6cengMOE.pt', type=str)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--pth", default='predata/mixmsk.jbl', type=str)
    parser.add_argument("--pth2", default='predata/preM2T.jbl', type=str)
    parser.add_argument("--pth3", default='predata/spat.jbl', type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=4, type=int)
    parser.add_argument("--epoch", default=1, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--global_step", default=500001, type=int)
    parser.add_argument("--step_pre", default=0, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)


