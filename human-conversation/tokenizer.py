from pathlib import Path
import numpy as np
import pandas as pd

file_path = Path(__file__).resolve().parent / "one_piece.txt"
data = file_path.read_text(encoding="utf-8")
words = data.split()
list=[]

for word in words[0:10]:
    for i in word:
        list.append(ord(i))

list= np.array(list)
print(list)