#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, List
from itertools import combinations
from tqdm import tqdm
from vllm import LLM, SamplingParams

# ----------------------------
# Hardcoded paths and settings
# ----------------------------

dataset = 'us_Home'
RELATION = "compared"
# RELATION = "alsoViewed"
# RELATION = "boughtTogether"
# RELATION = "alsoBought"
MODEL_PATH = "/home/stlamb/Phi-4-AutoRound-GPTQ-4bit"
METADATA_JSONL = f"{dataset.lower()}/metadata_{dataset}.json"   # one JSON per line
dataset_dir = f"{dataset.lower()}/"
INPUT_LINES = f"{dataset_dir}{RELATION}.filtered.txt"             # one space-separated ASIN list per line
OUTPUT_FILE = f"{dataset_dir}{RELATION}.filtered.substitute_counts.txt"
BATCH_SIZE = 12  # prompts per vLLM generate() call

# ----------------------------
# vLLM client and sampling
# ----------------------------


# ----------------------------
# Helpers
# ----------------------------
def load_title_map(jsonl_path: str) -> Dict[str, str]:
    """Build {asin: title} dict from metadata JSONL (one JSON per line)."""
    title_map: Dict[str, str] = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                asin = obj.get("asin")
                title = obj.get("title", "").strip()
                if asin and title:
                    title_map[asin] = title
            except Exception:
                # Skip bad lines silently
                continue
    return title_map

def calculate_combinations(k):
    """
    Calculates C(k, 2) using the simplified formula (k * (k - 1)) / 2.
    """
    if k < 2:
        return 0  # You can't choose 2 items from a set smaller than 2
    
    # Use integer division // to ensure the result is an integer
    return (k * (k - 1)) // 2


def build_prompt(asins: List[str], titles: Dict[str, str]) -> str:
    """Create a deterministic prompt asking for the number of substitute pairs, without including ASINs."""
    item_lines = []
    for idx, a in enumerate(asins, start=1):
        title = titles.get(a, "[TITLE MISSING]")
        item_lines.append(f"{idx}) {title}")

    k = len(asins)
    prompt = f"""You are given k Amazon Electronics items. For each item, only the product title is provided.

A "substitute pair" means the two items serve the same main function and a typical customer would choose one instead of the other in a single purchase for that purpose.
Do not count pairs where items are complementary (e.g., a camera and a lens) as substitutes.
Capacity/color variants of the same product line are substitutes. Ignore seller/bundle differences.

Items:
{chr(10).join(item_lines)}

Question: Among all pairs of these k items (C(k, 2) pairs), how many are substitute pairs?

Return only a single integer with no words.
k = {k}
"""
    return prompt

def build_pairwise_prompt(item_pair, dataset, mode = 's'):
    category = {'us_electronics': 'in electronics on Amazon', 
                'us_office':'in office products on Amazon',
                'us_home':'in home & kitchen category on Amazon'}
    if dataset not in category:
        raise ValueError("Missing given prompt for the dataset")
    relation = 'substitutes' if mode == 's' else 'complements'
    prompt = f"""
    Question: Assume two items can be substitutes/complements/irrelevant. Are the item pair below are {relation} {category[dataset]}?

    Item Pair:
    {chr(10).join(item_pair)}

    Return only YES or NO.
    """
    return prompt

def build_prompt_strict(asins: List[str], titles: Dict[str, str]) -> str:
    """Stricter fallback prompt for retries."""
    item_lines = []
    for idx, a in enumerate(asins, start=1):
        title = titles.get(a, "[TITLE MISSING]")
        item_lines.append(f"{idx}) {title}")

    prompt = f"""Task: Count substitute pairs among the items below.

Substitute pair = two items that perform the same main function, so a customer would typically buy ONE or the OTHER for that function.
Do NOT count complementary items. Variants (capacity/color) ARE substitutes. Ignore seller/bundle differences.

Items:
{chr(10).join(item_lines)}

Output format requirement:
- Print ONE integer ONLY.
- No words. No punctuation. No spaces. No extra lines.

Now print the integer:
"""
    return prompt


def extract_first_integer(text: str) -> int:
    """Parse the first integer from model output; default to 0 if none."""
    nums = re.findall(r"-?\d+", text)
    if not nums:
        return 0
    try:
        return int(nums[0])
    except Exception:
        return 0


def extract_first_yes_no(text: str) -> int:
    """
    Parse the first Yes/No (case-insensitive) from model output.
    Return 1 for Yes, 0 for No, default 0 if none found.
    """
    match = re.search(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
    if not match:
        return 0
    
    token = match.group(1).lower()
    return 1 if token == "yes" else 0

def read_input_lines(path: str) -> List[List[str]]:
    """Read space-separated ASIN lists from file."""
    groups: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            groups.append(line.split())
    return groups


def write_results(path: str, groups: List[List[str]], counts: List[int]) -> None:
    """Write '<asin...>\t<count>' per line."""
    assert len(groups) == len(counts)
    with open(path, "w", encoding="utf-8") as out:
        for asins, c in zip(groups, counts):
            out.write(f"{' '.join(asins)}\t{c}\n")

def vllm_batch_output(instruction_list, llm_client, sampling_params):
        messages_list = [[
                    {"role": "system", "content": "Answer only 'Yes' or 'No'."},
                    {"role": "user", "content": prompt}] 
                    for prompt in instruction_list]
        outputs = llm_client.chat(messages_list, sampling_params, use_tqdm=False)
        results = [req.outputs[0].text for req in outputs]
        return results


def query_llm_batch_sub(batch_prompts, all_counts,llm_client, sampling_params):
    outputs = vllm_batch_output(batch_prompts, llm_client, sampling_params)
    for out in outputs:
        cnt = extract_first_yes_no(out)
        all_counts.append(cnt)
    return all_counts
# ----------------------------
# Main
# ----------------------------

def calculate_sub_sum(group, item_pair_dict, item_pair_map, all_counts_sub, all_counts_com):
    item_pairs = list(combinations(group, 2))
    sub, com, ttl = 0, 0, len(item_pairs)
    for item_p in item_pairs:
        if item_p[::-1] in item_pair_dict:
            sub += all_counts_sub[item_pair_map[item_p[::-1]]]
            com += all_counts_com[item_pair_map[item_p[::-1]]]
        elif item_p in item_pair_dict:
            sub += all_counts_sub[item_pair_map[item_p]]
            com += all_counts_com[item_pair_map[item_p]]
        else:
            print(f'{item_p} not in item_pair_dict')
    return sub, com, ttl

def main():
    mode = 1
    dataset = dataset_dir.split('/')[0]
    print(f'TARGET DATASET: {dataset}')
    # Sanity checks
    if not Path(METADATA_JSONL).exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_JSONL}")
    if not Path(INPUT_LINES).exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_LINES}")

    print("Loading metadata titles...")
    title_map = load_title_map(METADATA_JSONL)
    print(f"Loaded {len(title_map):,} ASIN titles.")

    print("Reading input groups...")
    asin_groups = read_input_lines(INPUT_LINES)
    print(f"Total groups: {len(asin_groups):,}")
    sampling_params = SamplingParams(
        temperature=0.0,   # STRICT deterministic
        top_p=1.0,         # no probability truncation
        top_k=-1,          # allow full vocab (not needed, but fine)
        max_tokens=8,      # enough for "yes" / "no"
        seed=2025,
    )

    llm_client = LLM(
        model=MODEL_PATH,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
    )
    if mode == 0:
        # Build prompts
        prompts = [build_prompt(group, title_map) for group in asin_groups]

        # Run in batches
        all_counts: List[int] = []
        print("Querying LLM...")
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i : i + BATCH_SIZE]
            outputs = vllm_batch_output(batch_prompts, llm_client, sampling_params)
            # vLLM returns a list of RequestOutput; each has .outputs (list of candidates)
            for out in outputs:
                cnt = extract_first_integer(out)
                all_counts.append(cnt)

            # Optional progress print
            done = min(i + BATCH_SIZE, len(prompts))
            print(f"Processed {done}/{len(prompts)}")

        print("Writing results...")
        write_results(OUTPUT_FILE, asin_groups, all_counts)
        print(f"Done. Results saved to: {OUTPUT_FILE}")
    elif mode == 1:
        item_pair_dict = {}
        idx = 0
        prompts_list_sub, prompts_list_com = [], []
        item_pair_map = {}
        for group in asin_groups:
            item_pairs = list(combinations(group, 2))
            for item_p in item_pairs:
                if (item_p[0] not in title_map) or (item_p[1] not in title_map):
                    if (item_p[0] not in title_map):
                        print(f'{item_p[0]} not in meta json.')
                    if (item_p[1] not in title_map):
                        print(f'{item_p[1]} not in meta json.')
                    continue
                elif (item_p[::-1] in item_pair_dict) or (item_p in item_pair_dict):
                    continue
                else:
                    item_pair_dict[item_p] = -1
                    item_pair_map[item_p] = idx
                    idx += 1
                    item_title_p = (title_map[item_p[0]], title_map[item_p[1]])
                    pairwise_prompt_sub = build_pairwise_prompt(item_title_p, dataset, mode = 's')
                    pairwise_prompt_com = build_pairwise_prompt(item_title_p, dataset, mode = 'c')
                    prompts_list_sub.append(pairwise_prompt_sub)
                    prompts_list_com.append(pairwise_prompt_com)
        print("Querying LLM...")
        all_counts_sub, all_counts_com = [], []
        for i in tqdm(range(0, len(prompts_list_sub), BATCH_SIZE)):
            # print(prompts_list_sub[i : i + BATCH_SIZE])
            all_counts_sub = query_llm_batch_sub(prompts_list_sub[i : i + BATCH_SIZE], all_counts_sub, llm_client, sampling_params)
            all_counts_com = query_llm_batch_sub(prompts_list_com[i : i + BATCH_SIZE], all_counts_com, llm_client, sampling_params)
        with open(OUTPUT_FILE, "w") as f:
            for group in asin_groups:
                sub, com, ttl = calculate_sub_sum(group, item_pair_dict, item_pair_map, all_counts_sub, all_counts_com)
                line = ' '.join(group) + f" {sub} {com} {ttl}"
                f.write(line + "\n")




if __name__ == "__main__":
    main()
