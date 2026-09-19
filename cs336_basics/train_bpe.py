import os
import regex as re

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
    merges: list[tuple[bytes, bytes]] = []
    special_pretokens: list[str] = []
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
        if special_tokens is not None and special_tokens != []:
            PAT = "|".join(re.escape(special_token) for special_token in special_tokens)
            last = 0
            for i in re.finditer(PAT, text):
                special_pretokens.append(text[last:i.start()])
                last = i.end()
            if last < len(text):
                special_pretokens.append(text[last:])
        else:
            special_pretokens = [text]
        pretokens: list[str] = []
        for token in special_pretokens:
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            for i in re.finditer(PAT, token):
                pretokens.append(token[i.start():i.end()])

        totaltokens: dict[tuple[bytes, ...], int] = {}
        for token in pretokens:
            raw_bytes = token.encode("utf-8")
            tokens = []
            for b in raw_bytes:
                tokens.append(bytes([b]))
            if tuple(tokens) not in totaltokens:
                totaltokens[tuple(tokens)] = 1
            else:
                totaltokens[tuple(tokens)] += 1
        while len(vocab) < vocab_size:
            count: dict[tuple[bytes, bytes], int] = {}
            for tokens, freq in totaltokens.items():   
                for i in range(len(tokens) - 1):
                    if (tokens[i], tokens[i + 1]) in count:
                        count[(tokens[i], tokens[i + 1])] += freq
                    else:
                        count[(tokens[i], tokens[i + 1])] = freq
            #max_count = max(count.values(), default=0)    
            max_pair = max(count, key=lambda pair: (count[pair], pair), default=None) 
            if max_pair is None:
                break
            merges.append(max_pair)
            vocab[len(vocab)] = max_pair[0] + max_pair[1]
            next_totaltokens: dict[tuple[bytes, ...], int] = {}
            for tokens, freq in totaltokens.items():
                next_tokens = []
                i = 0
                while i < len(tokens):
                    if i != len(tokens) - 1 and (tokens[i], tokens[i + 1]) == max_pair:
                        next_tokens.append(tokens[i] + tokens[i + 1])
                        i += 2
                    else:
                        next_tokens.append(tokens[i])
                        i += 1
                if tuple(next_tokens) not in next_totaltokens:
                    next_totaltokens[tuple(next_tokens)] = freq
                else:
                    next_totaltokens[tuple(next_tokens)] += freq
            totaltokens = next_totaltokens

    return vocab, merges