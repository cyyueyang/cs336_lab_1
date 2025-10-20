import heapq
import time
from typing import List, Tuple, Dict, DefaultDict, Any
from collections import defaultdict
import mmap
import random
import regex
import re
import multiprocessing
import os
from tqdm import tqdm
import json
import psutil


"""
合并过程:
1. pair_positions   维护每个pair的position
2. positions_by_seq 维护每个pair在每个序列中的位置
3. 倒序处理
4. last_merged_pos 记录最后一次合并的位置
5. 更新左侧字节对
6. 更新右侧字节对
"""
GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def load_and_sample_data(file_path: str,
                         sample_size: int,
                         special_token: str = "<|endoftext|>"
                         ) -> str:
    try:
        with open(file_path, "r+", encoding="utf-8", errors="ignore") as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                documents = []
                start = 0
                while start < len(mm):
                    end = mm.find(special_token.encode("utf-8"), start)

                    if end == -1:
                        doc = mm[start:].decode("utf-8", errors="replace").strip()
                        if doc:
                            documents.append(doc)
                        break
                    else:
                        doc = mm[start:end].decode("utf-8", errors="replace").strip()
                        if doc:
                            documents.append(doc)
                        start = end + len(special_token.encode("utf-8"))
                if len(documents) > sample_size:
                    documents = random.sample(documents, sample_size)
                return special_token.join(documents)
    except Exception as e:
        raise IOError(f"Error opening {file_path}: {e}")

def gpt_bytes_to_unicode_local() -> Dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]

    n = 0
    for i in range(256):
        if i not in bs:
            bs.append(i)
            cs.append(n + 256)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def pre_tokenizer_document(doc: str, bytes_to_unicode_map: Dict[int, str]) -> List[List[str]]:
    tokens = regex.findall(GPT2_PAT, doc, flags=regex.UNICODE)
    sequences = []
    for token in tokens:
        token_unicode = "".join([bytes_to_unicode_map[b] for b in token.encode("utf-8")])
        sequences.append(list(token_unicode))
    # 单个文档的所有token序列 eg [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']
    return sequences

def parallel_pre_tokenizer(documents: List[str],
                           num_processes: int,
                           bytes_to_unicode_map: Dict[int, str]
                           ) -> List[List[str]]:
    if num_processes == 1:
        return [seq for doc in documents for seq in pre_tokenizer_document(doc, bytes_to_unicode_map)]

    with multiprocessing.Pool(num_processes,
                              initializer=init_worker,
                              initargs=(bytes_to_unicode_map, )) as pool:

        # result: List[List[List[str]]]
        # 最内层: List[str]: 单个token的字符列表 eg ['h', 'e', 'l', 'l', 'o']
        # 中间层: List[List[str]] 单个文档的所有token序列 eg [['h', 'e', 'l', 'l', 'o'], ['w', 'o', 'r', 'l', 'd']
        # 最外层: List[List[List[str]]] 所有单个文档的list
        result = list(tqdm(
            pool.imap(pre_tokenizer_worker, documents, chunksize=50),
            total=len(documents),
            desc="Pre-tokenizing documents",
            mininterval=1
        ))
        # 所有文档的token序列集合
        return [seq for doc_sequences in result for seq in doc_sequences]

global_worker_byte_map = None
def init_worker(byte_map: Dict[int, str]):
    global global_worker_byte_map
    global_worker_byte_map = byte_map

def pre_tokenizer_worker(doc: str):
    return pre_tokenizer_document(doc, bytes_to_unicode_map=global_worker_byte_map)

class BPEIndex:

    def __init__(self, sequences: List[List[str]]):
        self.sequences = sequences
        self.pair_counts: DefaultDict[Tuple[str, str], int] = defaultdict(int)
        self.pair_positions: DefaultDict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list) # [第几个预分词token， 第几个预分词token中的位置]
        self.heap = []
        self.heap_entries: Dict[Tuple[str, str], Any] = {} # 追踪堆中的条目 方便更新

        # init pair_counts, pair_positions 防止跨文档合并
        for seq_idx, seq in enumerate(self.sequences):
            for pos in range(len(seq) - 1):
                pair = (seq[pos], seq[pos + 1])
                self.pair_counts[pair] += 1
                self.pair_positions[pair].append((seq_idx, pos))

        for pair, count in self.pair_counts.items():
            if count > 1:
                entry = [-count, pair]
                # 堆 方便获取最大频率的相邻字节
                heapq.heappush(self.heap, entry)
                # 空间换时间 方便修改堆中的(-count, pair)
                self.heap_entries[pair] = entry

    def get_most_frequent_pairs(self) -> Tuple[str, str]:
        # bpe 算法 会影响当前合并序列 左右两个序列的数量，所以要对堆顶进行多个判断
        while self.heap:
            neg_count, pair = self.heap[0]
            # 最高频次 但是已经被合并过了
            if pair not in self.heap_entries:
                heapq.heappop(self.heap)
                continue

            current_count = self.pair_counts.get(pair, 0)

            if -neg_count == current_count and current_count > 1:
                return pair
            # 处理 计数不匹配或者count=1 的情况
            heapq.heappop(self.heap)

            if pair in self.heap_entries:
                del self.heap_entries[pair]

        return None

    def _update_pair_count(self, pair: Tuple[str, str], delta: int):
        if delta == 0:
            return

        if pair not in self.pair_counts:
            self.pair_counts[pair] = 0

        new_count = self.pair_counts[pair] + delta
        self.pair_counts[pair] = new_count

        if new_count < 0:
            new_count = 0
            self.pair_counts[pair] = new_count

        if pair in self.heap_entries and self.heap_entries[pair] is not None:
            self.heap_entries[pair][0] = -new_count
            heapq.heapify(self.heap)
        elif new_count > 1:
            entry = [-new_count, pair]
            heapq.heappush(self.heap, entry)
            self.heap_entries[pair] = entry

    def _add_position(self, pair: Tuple[str, str], seq_idx: int, pos: int):
        self.pair_positions[pair].append((seq_idx, pos))

    def merge_pair(self, pair: Tuple[str, str], new_token: str) -> int:
        if pair not in self.pair_positions or not self.pair_positions[pair]:
            return 0

        positions_by_seq = defaultdict(list)
        for seq_idx, pos in self.pair_positions[pair]:
            positions_by_seq[seq_idx].append(pos)

        merge_count = 0

        for seq_idx, position in positions_by_seq.items():
            seq = self.sequences[seq_idx]
            position.sort(reverse=True)

            last_merged_pos = -2

            for pos in position:
                # 检查是否被前面的合并影响
                if pos >= len(seq) - 1 or pos <= last_merged_pos:
                    continue
                if seq[pos] != pair[0] or seq[pos + 1] != pair[1]:
                    continue

                seq[pos] = new_token

                del seq[pos + 1]
                merge_count += 1
                last_merged_pos = pos

                # 更新左侧
                if pos > 0:
                    left_pair = (seq[pos - 1], pair[0])
                    self._update_pair_count(left_pair, delta=-1)
                    new_left_pair = (seq[pos - 1], new_token)
                    self._update_pair_count(new_left_pair, delta=1)
                    self._add_position(new_left_pair, seq_idx, pos - 1)

                if pos < len(seq) - 1:
                    right_pair = (pair[1], seq[pos + 1])
                    self._update_pair_count(right_pair, delta=-1)
                    new_right_pair = (new_token, seq[pos + 1])
                    self._update_pair_count(new_right_pair, delta=1)
                    self._add_position(new_right_pair, seq_idx, pos)

        if pair in self.pair_counts:
            del self.pair_counts[pair]
        if pair in self.pair_positions:
            del self.pair_positions[pair]
        if pair in self.heap_entries:
            self.heap_entries[pair] = None

        return merge_count

def run_train_bpe(
        input_path: str,
        vocab_size: int,
        special_tokens: List[str] = ["<|endoftext|>"],
        num_processes: int = 8,
        sample_size: int = 22000,
        **kwargs,
):
    base_vocab_size = len(special_tokens) + 256

    if vocab_size < base_vocab_size:
        raise ValueError("Vocab size is too small")

    bytes_to_unicode_map = gpt_bytes_to_unicode_local()
    unicode_to_bytes_map = {v: bytes([k]) for k, v in bytes_to_unicode_map.items()}

    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256
    existing_bytes = set(vocab.values())

    for special_token in special_tokens:
        special_token = special_token.encode("utf-8")
        if special_token not in existing_bytes and len(vocab) < vocab_size:
            vocab[next_token_id] = special_token
            next_token_id += 1
            existing_bytes.add(special_token)

    print(f"from {input_path} sample {sample_size} documents:")
    text = load_and_sample_data(input_path, sample_size, special_token=special_tokens[0])

    escaped_tokens = [re.escape(special_token) for special_token in special_tokens]
    split_pattern = "|".join(escaped_tokens)
    documents = [part for part in re.split(split_pattern, text) if part] # List[str] 过滤掉空字符串

    sequences = parallel_pre_tokenizer(documents, num_processes=num_processes, bytes_to_unicode_map=bytes_to_unicode_map)
    print(f"pre_tokenize over get {len(sequences)} tokens")
    print("build bpe index")
    bpe_index = BPEIndex(sequences=sequences)
    merges = []
    vocab_progress = len(vocab)
    total_merges = vocab_size - vocab_progress

    print("BPE Trainging")
    print("total merges: ", total_merges)
    progress_bar = tqdm(total=total_merges, desc="Training BPE", unit="merges", mininterval=0.5)

    while vocab_progress < vocab_size:
        best_pair = bpe_index.get_most_frequent_pairs()
        if best_pair is None:
            print("early stopping")
            break

        new_token_str = best_pair[0] + best_pair[1]
        p1_bytes = unicode_to_bytes_map[best_pair[0]]
        p2_bytes = unicode_to_bytes_map[best_pair[1]]

        new_token_bytes = p1_bytes + p2_bytes
        merge_count = bpe_index.merge_pair(best_pair, new_token_str)

        if merge_count == 0:
            continue

        if new_token_bytes not in existing_bytes:
            vocab[next_token_id] = new_token_bytes
            next_token_id += 1
            existing_bytes.add(new_token_bytes)
            merges.append((p1_bytes, p2_bytes))
            vocab_progress += 1
            progress_bar.update(1)

        unicode_to_bytes_map[new_token_str] = new_token_bytes

    progress_bar.close()

    return vocab, merges

def evaluate_tokenizer(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], test_text: str) -> Tuple[float, float]:
    print("evaluating tokenizer ...")
    sample_text = test_text[:200] + "..." if len(test_text) > 200 else test_text

    print(f"test text: {sample_text}")
    unique_tokens = set(vocab.values())
    print(f"unique tokens: {len(unique_tokens)}")
    print(f"vocab size: {len(vocab)}")
    print(f"merge size: {len(merges)}")

def save_vocab_and_merges(vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    print("saving vocab and merges ...")
    vocab_str = {idx: token.encode("utf-8") for idx, token in vocab}
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, ensure_ascii=False, indent=4)
    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            part1 = merge[0].decode("utf-8")
            part2 = merge[1].decode("utf-8")
            f.write(f"{part1} {part2}")

if __name__ == "__main__":
    config = {
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>", "<pad>", "<unk>"],
        "num_processes": 8,
        "sample_size": 22000,
    }

    train_path = r"../data/TinyStoriesV2-GPT4-train.txt"
    valid_path = r"../data/TinyStoriesV2-GPT4-valid.txt"

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(valid_path):
        raise FileNotFoundError(valid_path)

    start_time = time.time()
    print("start training...")
    train_vocab, train_merges = run_train_bpe(train_path, **config)
    print("finished training.")
    print(f"use time: {time.time() - start_time}")

    print("small demo evaluate...")
    valid_config = config.copy()
    valid_config["sample_size"] = int(config["sample_size"] * 0.1)
    valid_vocab, valid_merges = run_train_bpe(valid_path, **valid_config)

    print("training results...")
    print(f"train vocab size: {len(train_vocab)}")
    print(f"valid vocab size: {len(valid_vocab)}")
    print(f"train merge size: {len(train_merges)}")
    print(f"valid merge size: {len(valid_merges)}")

    train_tokens = set(train_vocab.values())
    valid_tokens = set(valid_vocab.values())
    overlap = train_tokens & valid_tokens
    print(f"overlap rate: {len(overlap) / len(train_tokens)}")
    with open(valid_path, "r", encoding="utf-8") as f:
        valid_text = f.read(1000)
    evaluate_tokenizer(train_vocab, train_merges, valid_text)

    output_dir = "./bpe_output"
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocab.json")
    merge_path = os.path.join(output_dir, "merges.txt")

    save_vocab_and_merges(train_vocab, train_merges, vocab_path, merge_path)
    print("save vocab and merges finished.")
    print(f"vocab saved to {vocab_path}")
    print(f"merges saved to {merge_path}")

    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / 1024 ** 2
    print(f"memory usage: {mem_usage:.2f} MB")















