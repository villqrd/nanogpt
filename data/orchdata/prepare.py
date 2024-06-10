from collections import defaultdict
import itertools
import json
import os
import numpy as np
from pathlib import Path
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

FILE = Path(__file__).parent

PRICES = {  # per million tokens USD https://www.together.ai/pricing
    "bigcode--starcoder2-3b": 0.1,
    "bigcode--starcoder2-7b": 0.2,
    "bigcode--starcoder2-15b": 0.3,
    "mistralai--Codestral-22B-v0.1": 0.8,
    "smallcloudai--Refact-1_6B-fim": 0.1,
}


def compute_signal(y, prices):
    y = (y / np.log(1+prices)).softmax(dim=1)
    return y.max(dim=1).values


def prep(max_length=64, min_signal_th=0.999):
    model_id = "mistralai/Codestral-22B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        truncation_side="left",
        padding_side="right",
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    indir = Path("/home/villqrd/d1/")
    ds_keep = ["orch_func_complete", "orch_next_line"]
    all_prompts = defaultdict(list)
    all_scores = defaultdict(list)
    model_dirs = sorted(indir.iterdir())
    prices = []

    for ix_ds, ds_name in enumerate(ds_keep):
        for model_dir in model_dirs:
            scores_path = model_dir / f"scores_{ds_name}.json"
            assert scores_path.exists()
            prompt_path = model_dir / f"prompt_{ds_name}.json"
            assert prompt_path.exists()
            all_prompts[ds_name].append(json.loads(prompt_path.read_text()))
            scores = json.loads(scores_path.read_text())
            all_scores[ds_name].append(scores)
            if ix_ds == 0:
                prices.append(PRICES[model_dir.name])
            print(f"Found {ds_name} for {model_dir}")
        assert all(
            all_prompts[ds_name][0] == prompt for prompt in all_prompts[ds_name][1:]
        )
    prices = np.array(prices, np.float16)
    all_prompts = [all_prompts[ds_name][0] for ds_name in ds_keep]
    all_prompts = list(itertools.chain(*all_prompts))
    all_scores = [
        np.hstack(
            [np.array(scores, np.float16)[:, None] for scores in all_scores[ds_name]]
        )
        for ds_name in ds_keep
    ]
    all_scores = np.vstack(all_scores)
    assert len(all_prompts) == all_scores.shape[0]
    signals = compute_signal(torch.from_numpy(all_scores), prices)
    high_signals = signals > min_signal_th
    all_scores = all_scores[high_signals]
    all_prompts = [
        prompt for prompt, high_signal in zip(all_prompts, high_signals) if high_signal
    ]
    assert len(all_prompts) == all_scores.shape[0]

    all_prompts_tk = []
    start_ixs = []
    end_ixs = []
    start_ix = 0
    for prompt in tqdm(all_prompts):
        input_ids = tokenizer(
            prompt, truncation=True, padding="max_length", max_length=max_length
        )["input_ids"]
        try:
            new_line_ix = input_ids.index(781)
            for i in range(new_line_ix + 1):
                input_ids[i] = tokenizer.pad_token_id
        except ValueError:
            print("careful, no new line token in prompt")
        start_ixs.append(start_ix)
        end_ixs.append(start_ix + len(input_ids))
        start_ix += len(input_ids)
        all_prompts_tk.append(input_ids)
    all_prompts_tk = np.concatenate(all_prompts_tk).astype(np.uint16)
    start_ixs = np.array(start_ixs, dtype=np.uint32)
    end_ixs = np.array(end_ixs, dtype=np.uint32)
    indices = np.hstack([start_ixs[:, None], end_ixs[:, None]])

    n = indices.shape[0]
    train_indices = indices[: int(n * 0.9)]
    val_indices = indices[int(n * 0.9) :]
    print(f"Train: {train_indices.shape[0]}, Val: {val_indices.shape[0]}")

    all_scores.tofile(os.path.join(os.path.dirname(__file__), "scores.bin"))
    all_prompts_tk.tofile(os.path.join(os.path.dirname(__file__), "prompts.bin"))
    prices.tofile(
        os.path.join(os.path.dirname(__file__), "prices.bin")
    )
    train_indices.tofile(os.path.join(os.path.dirname(__file__), "train_indices.bin"))
    val_indices.tofile(os.path.join(os.path.dirname(__file__), "val_indices.bin"))


def get_batch(tokenizer, split, batch_size, device, n_models=5, max_length: int = None):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    data = np.memmap(
        os.path.join(os.path.dirname(__file__), "prompts.bin"),
        dtype=np.uint16,
        mode="r",
    )
    prices = torch.from_numpy(
        np.memmap(
            os.path.join(os.path.dirname(__file__), "prices.bin"),
            dtype=np.float16,
            mode="r",
        )
    ).to(torch.float32)

    scores = np.memmap(
        os.path.join(os.path.dirname(__file__), "scores.bin"),
        dtype=np.float16,
        mode="r",
    )
    if split == "train":
        indices = np.memmap(
            os.path.join(os.path.dirname(__file__), "train_indices.bin"),
            dtype=np.uint32,
            mode="r",
        )
    elif split == "val":
        indices = np.memmap(
            os.path.join(os.path.dirname(__file__), "val_indices.bin"),
            dtype=np.uint32,
            mode="r",
        )
    else:
        raise ValueError(f"Unknown split: {split}")
    indices = indices.reshape((-1, 2))
    scores = scores.reshape((-1, n_models))
    ix = torch.randint(indices.shape[0], (batch_size,))
    # ix = torch.randint(4, 5, (batch_size,))
    # ix = torch.arange(batch_size*4) + 8
    batch_indices = indices[ix].reshape(-1, 2)
    x = [
        torch.from_numpy(data[start:end].astype(np.int64))
        for start, end in batch_indices
    ]
    # x = tokenizer.pad(x, return_tensors='pt', max_length=2)['input_ids']
    x = torch.stack(x)
    y = (
        torch.from_numpy(scores[ix].reshape((-1, n_models)).astype(np.float32))
        / (1 + prices).log()
    )
    y = y.softmax(dim=1)
    w = compute_signal(y, prices)
    if "cuda" in str(device):
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y, w = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
            w.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y, w = x.to(device), y.to(device), w.to(device)
    return x, y, None


if __name__ == "__main__":
    prep()

# model_id = "mistralai/Codestral-22B-v0.1"
# tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, padding=True, max_length=2)
# tokenizer.add_special_tokens({'pad_token': '<pad>'})

# print(get_batch("", "train", 12, "cpu", max_length=3)[0])
