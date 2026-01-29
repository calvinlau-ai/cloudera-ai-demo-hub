import os
import re
import random
import math
import time
from collections import deque
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from safetensors.torch import save_file
import wandb


@dataclass
class GRPOConfig:
    B: int = 8  # DeepSeek's values are B=8, G=16??
    G: int = 8
    mini_B: int = 8  # grad_acc_steps = B * G // mini_B
    clip_epsilon: float = 0.2  
    beta: float = 0.2  # ChatGPT recommended that beta be 0.2+
    num_epoch: int = 1
    lr: float = 2e-6  # lr: 5e-6 => 2e-5
    mu: int = 3
    max_grad_norm: float = 1.0
    kl_clip = False
    schedule_beta = True

    
class GRPODataSet:
    sys_prompt = '''A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
                    The assistant first thinks about the reasoning process in the mind and then provides the user
                    with the answer. The reasoning process and answer are enclosed within <think> </think> and
                    <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
                    <answer> answer here </answer>.'''
    sys_prompt = ' '.join(map(lambda line: line.strip(), sys_prompt.split()))
    # begin_of_text = tokenizer.convert_tokens_to_ids('<|begin_of_text|>')
    # start_header_id = tokenizer.convert_tokens_to_ids('<|start_header_id|>')
    
    def __init__(self, config, device=torch.device('cuda:0')):
        self.ds = load_dataset('openai/gsm8k', 'main')["train"]
        self.pos = 0
        self.chat_template = lambda line: [
            {"role": "sys", "content": GRPODataSet.sys_prompt},
            {"role": "user", "content": line}
        ]
        # self.chat_template_nosys = lambda line: [{"role": "user", "content": line}]
        self.config = config
        self.device = device

    @staticmethod
    def extract_answer_by_hash(text: str) -> str | None:
        # Extract thinking and answer parts
        if "####" not in text:
            return text, None
        else:
            think, answer = text.split("####", 1)
            return think.strip(), answer.strip()

    @staticmethod
    def format_answers(answer_text):
        think, answer = GRPODataSet.extract_answer_by_hash(answer_text)
        answer_output = f'<think>{think}</think> <answer>{answer}</answer>'
        # messages = [
        #     {"role": "sys", "content": sys_prompt},
        #     {"role": "user", "content": question},
        #     {"role": "assistant", "content": think_text}
        # ]
        # return tokenizer.apply_chat_template(messages, tokenize=False)[0]
        return answer_output, answer

    def target_outputs(self):
        B, G = self.config.B, self.config.G
        questions = self.ds[self.pos:self.pos+B]['question']
        answers = self.ds[self.pos:self.pos+B]['answer']
        answers = [GRPODataSet.format_answers(answer) for answer in answers]
        answer_texts = [ans[0] for ans in answers]
        tok = tokenizer(answer_texts, padding=True, padding_side='right', return_tensors='pt')
        tok = tok['input_ids']

        # 很多tokenizer会在句首自动加上bos_token_id，如<|begin_of_text|>等，需要去掉
        if tok[0, 0] == tokenizer.bos_token_id:
            tok = tok[:, 1:]

        ans = [ans[1] for ans in answers]
        ans = [x for x in ans for _ in range(G)]
        
        return tok.to(self.device), ans

    def get_first_eots(self, output_ids, prompt_len):
        '''Output shape: (B*G,)'''
        L = output_ids.size(1) - prompt_len
        tok_o = output_ids[:, prompt_len:]
        eot_id = tokenizer.convert_tokens_to_ids('<|eot_id|>')
        
        # 找到output_ids序列的结束位置，i.e. the 1st occurrance of eot_id
        # 先生成一个长度为num_output_sequences的1-D Tensor，记为first_eots，形状为(B*G, L)
        # 最终的结果在first_eots中
        first_eots = torch.full(tok_o.shape[:-1], L).to(self.device)
        mask = torch.full([tok_o.size(0)], False).to(self.device)
        
        for step in range(L):
            # 每个step表示一个token的输出
            step_toks = tok_o[:, step]
            _mask = (step_toks == eot_id)
            first_eots[_mask & ~(first_eots<step)] = step
            mask |= _mask

        return first_eots

    def get_completion_masks(self, output_ids, prompt_len):
        '''Output shape: (B*G, L)'''
        B_G, L = output_ids.shape
        L -= prompt_len
        first_eots = self.get_first_eots(output_ids, prompt_len)
        completion_mask = torch.arange(L, device=self.device).expand(B_G, L) < first_eots[:, None]
        return completion_mask
    
    def samples(self):
        B, G = self.config.B, self.config.G
        messages = [self.chat_template(line) for line in self.ds[self.pos:self.pos+B]['question']]
        prompts = tokenizer.apply_chat_template(messages, tokenize=False)
        for i, prompt in enumerate(prompts):
            s = '<|start_header_id|>assistant<|end_header_id|>'
            if not prompt.endswith(s):
                prompts[i] = prompt + s
        inputs = tokenizer(prompts, padding=True, padding_side='left', return_tensors='pt')
        # print(inputs.keys())
        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)
        policy_model.eval()
        
        with torch.no_grad():
            sample_out = policy_model.generate(
                **inputs,
                max_length=512,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.7,               # sensible defaults
                top_p=0.9,
                num_return_sequences=G-1,
                # output_scores=True,
                # return_dict_in_generate=True
            )

        # sample_out = sample_out.sequences
        targets, ans = self.target_outputs()
        targets = torch.concat((inputs['input_ids'], targets), dim=1)
        prompt_len = inputs['input_ids'].size(1)
        delta = sample_out.size(1) - targets.size(1)
        out = []
        
        if delta > 0:
            # 采样序列的输出的答案比较长
            targets = F.pad(targets, (0, delta), value=tokenizer.eos_token_id)
        elif delta < 0:
            # 标准答案比较长
            sample_out = F.pad(sample_out, (0, -delta), value=tokenizer.eos_token_id)
            
        for i in range(B):  # 原先为targets.size(1)，可能打错。调整为ChatGPT建议的B
            out.append(sample_out[i*(G-1):(i+1)*(G-1)])
            out.append(targets[i:i+1])

        self.pos += B
        output_ids = torch.cat(out, dim=0)
        return output_ids, ans, prompt_len


## Formatting functions
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(txt_o, answer) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    # q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in txt_o]
    # print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(txt_o, **kwargs) -> list[float]:
    # responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in txt_o]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(txt_o, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    # responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in txt_o]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(txt_o, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    # responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in txt_o]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1:
        count += 0.125
    if text.count("\n</think>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(txt_o, **kwargs) -> list[float]:
    # contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in txt_o]


reward_funcs = [
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
]


class GRPOTrainer:    
    def __init__(self, config, dataset, device=torch.device('cuda:0')):
        self.config = config
        self.dataset = dataset
        self.device = device

    def _compute_rewards(self, txt_o, answer):
        _rewards = [r(txt_o) for r in reward_funcs[:-1]]
        _rewards.append(correctness_reward_func(txt_o, answer))
    
        # NOTE: rewards的“每列”表示一个sequence的reward，因而需要对dim=0作累加
        _rewards = torch.Tensor(_rewards).sum(dim=0).to(self.device)
        return _rewards
    
    def _compute_rewards_and_advantages(self, txt_o, answer):
        """组内标准化优势计算"""
        B, G = self.config.B, self.config.G
        rewards = self._compute_rewards(txt_o, answer).view(B, G)
        
        mean = rewards.mean(dim=1, keepdim=True)
        std = rewards.std(dim=1, keepdim=True) + 1e-8
        adv = (rewards - mean) / std  # (B, G)
        
        return rewards, adv.view(B*G)
    
    @staticmethod
    def _compute_log_probs(model, input_ids, L, gather=True):
        # 如果需要no_grad的话，需要在调用该函数前进入torch.no_grad()上下文
        # print('input_ids:', input_ids.shape)
        logits = model(input_ids).logits[:, -L-1:-1, :]
        log_probs = logits.log_softmax(dim=-1)
        if gather:
            return log_probs.gather(dim=-1, index=input_ids[:, -L:, None]).squeeze(-1)
        else:
            return log_probs

    @staticmethod
    def _update_beta(beta, kl, target_kl=0.1, eta=0.05):
        ratio = kl.detach() / target_kl
        beta = beta * (1 + eta * (ratio - 1.0))
        beta = float(min(max(beta, 1e-3), 10.0))
        return beta
          
    @staticmethod
    def _compute_per_token_kl(ref_log_probs, per_token_log_probs, kl_clip=False):
        x = ref_log_probs - per_token_log_probs

        if kl_clip:
          # 因为在计算KL时仅通过每个T中logit值最大的token进行近似，因而容易产生log_probs的突刺
          # 在一个序列中log_probs的最大值不能超过次大值的2倍，以免产生突刺
          x_max = x.topk(k=2, dim=1).values[:, 1] * 2
          x_max = x_max[:, None]
          x = torch.clip(x, -x_max, x_max)

        # 这儿的KL实际上是反向 KL (Reverse KL) 的一阶近似
        per_token_kl = torch.exp(x) - (x) - 1
        per_token_kl = torch.clip(per_token_kl, 0, 3)
        return per_token_kl

    def _compute_loss(self, input_ids, L, old_log_probs, ref_log_probs, advantages, completion_mask):
        # This is the loss function summarized from the experiments above
        config = self.config
        # 当计算logits，包含prompt，随后去除prompt的部分
        per_token_log_probs = GRPOTrainer._compute_log_probs(policy_model, input_ids, L)
        # import code; code.interact(local=locals())
        input_ids = input_ids[:, -L:]

        clip_epsilon = config.clip_epsilon
        ratio = torch.exp(per_token_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon)
    
        per_token_kl = GRPOTrainer._compute_per_token_kl(ref_log_probs, per_token_log_probs, kl_clip=config.schedule_beta)
        beta = GRPOTrainer._update_beta(config.beta, per_token_kl.mean()) if config.schedule_beta else config.beta
        per_token_adv = torch.min(ratio * advantages[:, None], clipped_ratio * advantages[:, None]) - beta * per_token_kl
        total_adv = ((per_token_adv * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # 注意：loss = -total_adv即为需要优化函数
        return -total_adv, per_token_log_probs, per_token_kl.mean()

    def train(self):
        # Merge the experiment into the function
        config = self.config
        optim = torch.optim.AdamW(policy_model.parameters(), lr=config.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.9)
        mini_B = config.mini_B
        grad_acc_steps = config.B * config.G // mini_B
        
        for step in range(500):
            t0 = time.time()
            input_ids, ans, prompt_len = dataset.samples()
            L = input_ids.size(1) - prompt_len
            logging.info(f'Input sample shape: ({len(input_ids)}, {L})')
            first_eots = dataset.get_first_eots(input_ids, prompt_len)
            logging.info(f'Long sequence (first_eots>300): {torch.nonzero(first_eots>300, as_tuple=True)[0]}')
            
            input_txt = [tokenizer.decode(out_tok[:first_eots[i]])
                         for i, out_tok in enumerate(input_ids[:, prompt_len:])]
            rewards, advantages = self._compute_rewards_and_advantages(input_txt, ans)
            completion_mask = torch.arange(L, device=self.device).expand(input_ids.size(0), L) <= first_eots[:, None]  # (B*G, L)
            torch.cuda.empty_cache()
            
            logging.info(f'Computing old_log_probs and ref_log_probs')
            old_log_probs = []
            ref_log_probs = []
            for start in range(0, len(input_ids), mini_B):
                end = start + mini_B
                with torch.no_grad():
                    old_log_probs.append(GRPOTrainer._compute_log_probs(policy_model, input_ids[start:end], L))
                    ref_log_probs.append(GRPOTrainer._compute_log_probs(ref_model, input_ids[start:end], L))
            old_log_probs = torch.cat(old_log_probs, dim=0)
            ref_log_probs = torch.cat(ref_log_probs, dim=0)
            # torch.cuda.empty_cache()
            dt_ref = time.time() - t0

            logging.info(f'Computing new_log_probs and loss for mu={config.mu}')
            # if L > 340:
            #     # Reduce grad_acc batch_size to avoid CUDA OOM for long sequences
            #     logging.info('Reduce grad_acc batch_size to avoid CUDA OOM for long sequences')
            #     splits_in_mini_B = 2
            # else:
            #     splits_in_mini_B = 1
            
            policy_model.train()
            for grpo_iter in range(config.mu):
                t0 = time.time()
                optim.zero_grad()
                loss_acc = 0.0
                kl_acc = 0.0
                log_probs_acc = []
                
                for acc_step in range(grad_acc_steps):
                    start = mini_B * acc_step
                    end = start + mini_B
                    loss, per_token_log_probs, kl = self._compute_loss(
                        input_ids[start:end], L,
                        old_log_probs[start:end], ref_log_probs[start:end],
                        advantages[start:end], completion_mask[start:end]
                    )
                    kl_acc += kl
                    log_probs_acc.append(per_token_log_probs)
                    loss = loss / grad_acc_steps  # Gradient Accumulation
                    loss_acc += loss.detach()
                    loss.backward()
                    torch.cuda.empty_cache()

                norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                
                if loss_acc > 0.3 or math.isnan(loss_acc) or kl > 3:
                    save_file({
                            'input_ids': input_ids,
                            'rewards': rewards,
                            'per_token_log_probs': torch.cat(log_probs_acc, dim=0),
                            'old_log_probs': old_log_probs,
                            'ref_log_probs': ref_log_probs,
                            'advantages': advantages,
                            'completion_mask': completion_mask,
                        },
                        f'train/train-{run_no}/train-{run_no}-step-{step}.{grpo_iter}.safetensors'
                    )
                    # torch.save(policy_model.state_dict(), f"llama-3.2-1b-gsm8k-before-{step}.{grpo_iter}.pth")
                
                optim.step()
                scheduler.step()
                torch.cuda.synchronize()
                
                dt = (time.time() - t0 + dt_ref / config.mu) * 1000
                print(
                    [f'Step {step}.{grpo_iter}, loss={loss_acc:.4f}, '
                    f'reward={(rewards.mean()):.4f}, reward_std={(rewards.std()):.4f}, '
                    f'kl={(kl_acc/config.mu):.4f}, norm={norm:.4f}, dt={dt:.2f}ms'][0]
                )
                wandb.log({
                    'loss': loss_acc, 'reward': rewards.mean(),
                    'reward_std': rewards.std(), 'kl': kl_acc/config.mu,
                    'norm': norm, 'lr': scheduler.get_last_lr()[0], 'dt': dt
                })


random.seed(42)
torch.manual_seed(42)
model_path = '/home/cdsw/models/Llama-3.2-1B-Instruct'
device = torch.device('cuda:0')

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'
logging.info('Loading policy_model ..')
policy_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)
policy_model.model.embed_tokens.weight.requires_grad = False
# policy_model = torch.compile(policy_model)
logging.info('Loading ref_model ..')
ref_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(device)

config = GRPOConfig()
wandb.init(project="grpo", config=asdict(config))
run_no = wandb.run.name.split("-")[-1]
os.makedirs(f'train/train-{run_no}', exist_ok=True)

dataset = GRPODataSet(config)
trainer = GRPOTrainer(config, dataset)
trainer.train()
torch.save(policy_model.state_dict(), f"train/train-{run_no}/llama-3.2-1b-gsm8k.pth")
