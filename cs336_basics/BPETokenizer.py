from typing import Iterator, Iterable
import regex as re
import pickle

class BPETokenizer(object):

    def __init__(self, vocab, merges, special_tokens=None):
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.vocab = vocab
        self.merges = merges
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_bytes = [token.encode('utf-8') for token in self.special_tokens]
        self.bytes_to_token_id = {v:k for k,v in vocab.items()}
        for token_bytes in self.special_tokens_bytes:
            if token_bytes not in self.bytes_to_token_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_token_id[token_bytes] = new_id

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        vocab = pickle.load(open(vocab_filepath, "rb"))
        merges = pickle.load(open(merges_filepath, "rb"))
        special_tokens = special_tokens if special_tokens is not None else []
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer

    def encode(self, text: str) -> list[int]:
        tokens = []

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if part in self.special_tokens:
                tokens.append(self.bytes_to_token_id[part.encode('utf-8')])
            else:
                tokens.extend(self._tokenize(part))

        return tokens

    def _to_bytes_tuple(self, word):
        l = list(word.encode('utf-8'))
        l = [bytes([i]) for i in l]
        return tuple(l)

    def _tokenize(self, text: str):
        pre_tokens = []
        for m in re.finditer(self.PAT, text):
            sub_word = m.group(0)
            pre_tokens.append(sub_word)

        token_ids = []
        for token in pre_tokens:
            byte_tuple = self._to_bytes_tuple(token)
            merged = self._apply_merges(byte_tuple)
            token_ids.extend(self.bytes_to_token_id[b] for b in merged)
        return token_ids

    def _apply_merges(self, byte_tuple):
        word = list(byte_tuple)

        def get_pair(word):
            pairs = set()
            prev_word = word[0]
            for char in word[1:]:
                pairs.add((prev_word, char))
                prev_word = char
            return pairs

        pairs = get_pair(word)

        if not pairs:
            return word

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            first, second = bigram
            new_word = []
            i = 0

            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pair(word)

        return list(word)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        full_bytes = b"".join(self.vocab[id_val] for id_val in ids)
        return full_bytes.decode('utf-8', errors='replace')

