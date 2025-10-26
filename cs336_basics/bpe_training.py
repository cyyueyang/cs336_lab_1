import regex as re
import os
import mmap
from typing import List, Tuple, Dict, DefaultDict, Any
from collections import defaultdict
from multiprocessing import Pool, Manager
import json

def process_chunk(chunk):
    local_pre_tokens_cnt = defaultdict(int)
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def word_to_bytes_tuple(word):
        word_bytes_list = list(word.encode('utf-8'))
        word_bytes_list = [bytes([i]) for i in word_bytes_list]
        return tuple(word_bytes_list)

    for m in re.finditer(PAT, chunk):
        sub_word = m.group(0)
        local_pre_tokens_cnt[word_to_bytes_tuple(sub_word)] += 1

    return local_pre_tokens_cnt

def train_bpe(input_path: str,
              vocab_size: int,
              special_tokens: List[str],
              **kwargs):
    num_processes = 8
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_idx = 256

    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_idx] = token_bytes
            next_idx += 1

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = re.split("|".join(map(re.escape, special_tokens)), text)

    pre_tokens_cnt = defaultdict(int)
    if len(chunks) > 1:
        with Pool(num_processes) as p:
            results = p.imap_unordered(process_chunk, chunks)

            for result in results:
                for token, cnt in result.items():
                    pre_tokens_cnt[token] += cnt
    else:
        result = process_chunk(chunks[0])
        for token, cnt in result.items():
            pre_tokens_cnt[token] += cnt

    merges = []

    pair_counts = defaultdict(int)
    for token, cnt in pre_tokens_cnt.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            pair_counts[pair] += cnt

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # 找到出现次数最多的对
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        first, second = best_pair
        new_token = first + second
        vocab[next_idx] = new_token
        next_idx += 1
        merges.append((first, second))

        # 更新 pre_tokens_cnt 和 pair_counts
        new_pre_tokens_cnt = {}
        tokens_to_remove = set()

        for token, cnt in pre_tokens_cnt.items():
            # 检查token中是否包含连续的best_pair
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # 从pair_counts中减少旧token的所有对计数
                for i in range(len(token) - 1):
                    pair = (token[i], token[i + 1])
                    pair_counts[pair] -= cnt
                    if pair_counts[pair] == 0:
                        del pair_counts[pair]  # 清理零计数对，避免膨胀

                # 创建新token：合并best_pair
                new_pre_token = []
                i = 0
                len_token = len(token)
                while i < len_token:
                    if i < len_token - 1 and token[i:i + 2] == best_pair:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)

                # 向pair_counts增加新token的所有对计数
                for i in range(len(new_pre_token) - 1):
                    pair = (new_pre_token[i], new_pre_token[i + 1])
                    pair_counts[pair] += cnt  # 使用defaultdict，自动处理缺失键

                # 记录新token和移除旧token
                new_pre_tokens_cnt[new_pre_token] = new_pre_tokens_cnt.get(new_pre_token, 0) + cnt
                tokens_to_remove.add(token)
            else:
                # 未修改的token直接保留
                new_pre_tokens_cnt[token] = new_pre_tokens_cnt.get(token, 0) + cnt

        # 批量更新pre_tokens_cnt
        for token in tokens_to_remove:
            del pre_tokens_cnt[token]
        pre_tokens_cnt.update(new_pre_tokens_cnt)

    return vocab, merges

if __name__ == '__main__':

    def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str,
                              merges_path: str):
        vocab_str = {idx: token.decode('utf-8', errors='replace') for idx, token in vocab.items()}
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        with open(merges_path, 'w', encoding='utf-8') as f:
            for merge in merges:
                part1 = merge[0].decode('utf-8', errors='replace')
                part2 = merge[1].decode('utf-8', errors='replace')
                f.write(f"{part1} {part2}\n")

    owt_train_data = r"../data/owt_train.txt"
    owt_val_data = r"../data/owt_valid.txt"
    TinyStories_train_data = r"../data/TinyStoriesV2-GPT4-train.txt"
    TinyStories_val_data = r"../data/TinyStoriesV2-GPT4-valid.txt"

    print("start training...")

    owt_train_vocab, owt_train_merges = train_bpe(input_path=owt_train_data, vocab_size=32000, special_tokens=["<|endoftext|>", "<pad>", "<unk>"])
    save_vocab_and_merges(vocab=owt_train_vocab, merges=owt_train_merges, vocab_path="../output/owt_train_vocab.json", merges_path="../output/owt_train_vocab.txt")
    print("owt_train_vocab merges finished")
    print("owt_train_vocab length:", len(owt_train_vocab))

    owt_val_vocab, owt_val_merges = train_bpe(input_path=owt_val_data, vocab_size=32000, special_tokens=["<|endoftext|>", "<pad>", "<unk>"])
    save_vocab_and_merges(vocab=owt_val_vocab, merges=owt_val_merges, vocab_path="../output/owt_val_vocab.json",merges_path="../output/owt_val_vocab.txt")
    print("owt_val_vocab merges finished")
    print("owt_train_vocab length:", len(owt_val_vocab))

    TinyStories_train_vocab, TinyStories_train_merges = train_bpe(input_path=TinyStories_train_data, vocab_size=10000, special_tokens=["<|endoftext|>", "<pad>", "<unk>"])
    save_vocab_and_merges(vocab=TinyStories_train_vocab, merges=TinyStories_train_merges, vocab_path="../output/TinyStories_train_vocab.json",merges_path="../output/TinyStories_train_vocab.txt")
    print("TinyStories_train_vocab merges finished")
    print("owt_train_vocab length:", len(TinyStories_train_vocab))

    TinyStories_val_vocab, TinyStories_val_merges = train_bpe(input_path=TinyStories_val_data, vocab_size=10000, special_tokens=["<|endoftext|>", "<pad>", "<unk>"])
    save_vocab_and_merges(vocab=TinyStories_val_vocab, merges=TinyStories_val_merges, vocab_path="../output/TinyStories_val_vocab.json", merges_path="../output/TinyStories_val_vocab.txt")
    print("TinyStories_val_vocab merges finished")
    print("owt_val_vocab length:", len(TinyStories_val_vocab))






