import re
import regex
import os
import mmap
from typing import List, Tuple, Dict, DefaultDict, Any
from collections import defaultdict

from numpy.ma.core import indices


def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: List[str],
              **kwargs):
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_idx = 256

    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    for token_bytes in special_token_bytes:
        vocab[next_idx] = token_bytes
        next_idx += 1

    with open(input_path, 'r', encoding="utf-8") as f:
        text = f.read()

    pre_tokens_cnt = defaultdict(int)
    def word_to_bytes_tuple(word):
        word_bytes_list = list(word.encode('utf-8'))
        word_bytes_list = [bytes([i]) for i in word_bytes_list]
        return tuple(word_bytes_list)

    chunks = regex.split("|".join(map(re.escape, special_tokens)), text)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            sub_word = m.group(0)
            pre_tokens_cnt[word_to_bytes_tuple(sub_word)] += 1

    merges = []
    while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair_counts[(token[i], token[i+1])] += cnt

        if not pair_counts:
            break

        max_pair_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_pair_count]
        best_pair = max(candidates)

        first, second = best_pair
        new_token = first + second
        vocab[next_idx] = new_token
        next_idx += 1
        merges.append((first, second))

        changes = []
        for token, cnt in pre_tokens_cnt.items():
            indices = [i for i in range(len(token) - 1) if token[i] == first and token[i + 1] == second]

            if indices:
                new_pre_token = []
                i = 0
                while i < len(token):
                    if token[i] == first and token[i + 1] == second:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1

                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        for old_token, new_token, cnt in changes:
            pre_tokens_cnt[new_token] = pre_tokens_cnt.get(new_token, 0) + cnt
            del pre_tokens_cnt[old_token]

    return vocab, merges
















