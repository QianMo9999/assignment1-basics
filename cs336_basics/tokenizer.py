import regex as re

class BPETokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = special_tokens
        else:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        self.reverse_vocab = {v: k for k, v in vocab.items()}
        self.rank: dict[tuple[bytes, bytes], int] = {}
        r = 0
        for m, n in self.merges:
            self.rank[(m, n)] = r
            r += 1

    def merge(self, tokens: list[bytes]) -> list[int]:
        i = 0
        next_tokens: list[bytes] = []
        while True:
            i = 0
            min_rank = len(self.rank)
            min_pair = None
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i+1]) in self.rank and self.rank[(tokens[i], tokens[i+1])] < min_rank:
                    min_rank = self.rank[tokens[i], tokens[i+1]]
                    min_pair = (tokens[i], tokens[i+1])
                i += 1
            if min_pair is None:
                break
            i = 0
            while i < len(tokens):
                if i != len(tokens)-1 and (tokens[i], tokens[i+1]) == min_pair:
                    next_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    next_tokens.append(tokens[i])
                    i += 1
            tokens = next_tokens
            next_tokens = []
        ids = []
        for t in tokens:
            ids.append(self.reverse_vocab[t])
        return ids

    def encode(self, string: str) -> list[int]:
        if string is None:
            return []
        
        # special tokens
        strList: list[tuple[bool, str]] = [] # True: special token, False: normal token
        last = 0
        if self.special_tokens is not None and self.special_tokens != []:
            pat = "|".join(re.escape(p) for p in self.special_tokens)
            for i in re.finditer(pat, string):
                strList.append((False, string[last:i.start()]))
                strList.append((True, i.group()))
                last = i.end()
            if last < len(string):
                strList.append((False, string[last:]))
        else:
            strList.append((False, string))

        # pretokenization and tokenization
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        ids: list[int] = []
        for is_special, s in strList:
            if not is_special:
                for i in re.finditer(PAT, s):
                    raw_bytes = i.group().encode("utf-8")
                    bytelist = []
                    for byte in raw_bytes:
                        bytelist.append(bytes([byte]))
                    ids.extend(self.merge(bytelist))
            else:
                ids.append(self.reverse_vocab[s.encode("utf-8")])

        return ids
    
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[id] for id in ids).decode("utf-8", errors="replace")

    def encode_iterable(self, iterable):
        for chunk in iterable:
            ids = self.encode(chunk)
            for id in ids:
                yield id
