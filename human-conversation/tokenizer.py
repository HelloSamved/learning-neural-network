from pathlib import Path

text_path = Path("one_piece.txt")
if not text_path.exists():
    text_path = Path("human-conversation") / "one_piece.txt"

text = text_path.read_text(encoding="utf-8")
words = text.split()

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

tokens = text.encode("utf-8") # raw bytes
tokens = list(map(int, tokens)) # convert to a list of integers in range 0..255 for convenience
print("length:", len(text))
print("first 100 tokens:", tokens[:100])
print("length:", len(tokens))

stats = get_stats(tokens)
print("unique byte pairs:", len(stats))
print("most common pairs:", sorted(stats.items(), key=lambda item: item[1], reverse=True)[:10])

vocab_size = 512 
num_merges = vocab_size - 256
ids = list(tokens) #Make a duplicate to avoid destroying our original list

merges = {}
new_id = []
for i in range(num_merges):
  if len(ids) < 2:
    break
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  new_id.append(idx)
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]

print(ids[:100])