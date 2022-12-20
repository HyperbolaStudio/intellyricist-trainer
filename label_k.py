from typing import *
import re
import hanlp
import time

pos:Callable[[List[str]], List[str]] = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
tok:Callable[[str], List[str]] = hanlp.load(hanlp.pretrained.tok.CTB9_TOK_ELECTRA_BASE)

input_file = open('./data/data_p.txt', 'r', encoding='utf-8')
output_file = open('./data/data_k2.txt', 'w', encoding='utf-8')

buffer: List[Tuple[Iterable[str], str]] = []
currentLyricId = ''

def bufferMapper(line: str, i: int):
    if(i + 3 < len(buffer)):
        labels = '/'.join(set([v for tags in map(lambda v: v[0], buffer[i:i+4]) for v in tags]))
        if(labels == ''): return None
        lyrics = '；'.join(map(lambda v: v[1], buffer[i:i+4]))
        prompt = ' = '.join((labels, lyrics))
        return '\t'.join((currentLyricId, prompt))
    else:
        return None

def getLabel(line: str):
    wordList = tok(line)
    return list(filter(
        lambda v: bool(v) and not ' ' in v,
        map(lambda word, tag: word if tag in ['NN', 'NT'] else None, wordList, pos(wordList))
    ))

counter = 0
lastTime = time.time()

startId = '1928358724'
stopId = '0'

started = False

while(True):
    line = input_file.readline()
    if(not line): break
    match = re.match('^========#([\d]+)', line)
    if(match):
        if(len(buffer)):
            output_file.write('\n'.join(
                filter(
                    lambda v: bool(v),
                    map(bufferMapper, buffer, range(len(buffer))),
                )
            ))
            counter += 1
            if(counter % 100 == 0):
                nowTime = time.time()
                print(counter, 'finished. ', 'Labeling last 100 entries in', nowTime - lastTime, 's')
                lastTime = nowTime
            buffer = []
        currentLyricId = line[match.start(1):match.end(1)]
        if(currentLyricId != startId and not started): continue
        else: 
            started = True
        if(currentLyricId == stopId): exit()
    else:
        if(not started): continue
        line = line.strip()
        buffer.append((getLabel(line),line))