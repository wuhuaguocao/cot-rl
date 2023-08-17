import os

import torch
from datasets import load_dataset

#############
from peft import LoraConfig
###########
import trlx
from trlx.data.default_configs import (
    ModelConfig,
    OptimizerConfig,
    PPOConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainConfig,
    TRLConfig,
)
import re




piqads = load_dataset('piqa', cache_dir='./dataset/piqa')
file_path = "./dataset/piqa/prompt1.txt"
with open(file_path, 'r') as file:
    file_contents = file.read()
prefix = file_contents

def wrapper(dic):
    string = "\nChoose the solution to achieve the goal and give answer in bracket:"
    key = list(dic.keys())
    value = list(dic.values())
    for i, j in zip(key[:-1], value[:-1]):
        string += "\n###" + i + ": " + j
    dic.update({'prompt': prefix + string + "\n### Thought:"})
    return dic

ds = piqads.map(wrapper)


def reward_fn(outputs, prompts, samples):
    rewards = []
    for output, prompt in zip(outputs, prompts):
        prompt_len = len(prompt)
        gen_output = output
        pattern = r"\[\d\]"
        result = re.findall(pattern=pattern, string=gen_output)
        if len(result) < 1:
            print("#"*300)
            print(output)
            print("#"*300)
            rewards.append(float(0))
            continue
        else:
            result = result[0]
        result = int(result.strip('[').strip(']'))
        pattern = r"goal:.*"
        split_sign = prefix[-300:]
        question = re.findall(pattern=pattern, string=prompt.split(split_sign)[-1])

        if len(question) < 1:

            rewards.append(float(0))
        else:
            question = question[0][6:]
            idx = ds['train']['goal'].index(question)
            label = ds['train']['label'][idx] + 1
            rewards.append(float(label == result))
    return rewards


def llama_config():
    return TRLConfig(
        train=TrainConfig(
            seq_length=1024,
            epochs=100,
            total_steps=6000,
            batch_size=1,
            checkpoint_interval=1000,
            eval_interval=100,
            pipeline="PromptPipeline",
            trainer="AcceleratePPOTrainer",
            save_best=True,
        ),
        model=ModelConfig(model_path="./model/llama-2-7b-hf", num_layers_unfrozen=2),
        tokenizer=TokenizerConfig(tokenizer_path="./model/llama-2-7b-hf", truncation_side="left"),
        optimizer=OptimizerConfig(
            name="adamw", kwargs=dict(lr=1.0e-5, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=1.0e-6)
        ),
        scheduler=SchedulerConfig(name="cosine_annealing", kwargs=dict(T_max=10000, eta_min=1.0e-5)),
        method=PPOConfig(
            name="PPOConfig",
            num_rollouts=128,
            chunk_size=4,
            ppo_epochs=4,
            init_kl_coef=0.05,
            target=6,
            horizon=10000,
            gamma=1,
            lam=0.95,
            cliprange=0.2,
            cliprange_value=0.2,
            vf_coef=1,
            scale_reward="ignored",
            ref_mean=None,
            ref_std=None,
            cliprange_reward=10,
            gen_kwargs=dict(
                max_new_tokens=256,
                top_k=0,
                top_p=1.0,
                do_sample=True,
            ),
        ),
    )


config = TRLConfig.update(llama_config().to_dict(), {})


###################################
config.model.peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    )
#####################################

if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = -1

trlx.train(
    reward_fn=reward_fn,
    prompts=ds['train']['prompt'][:-64],
    eval_prompts=ds['train']['prompt'][-64:],
    config=config,
)