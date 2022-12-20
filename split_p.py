from typing import *
import re

input_file = open('./data/data_p.txt', 'r', encoding='utf-8')
output_file = open('./data/data_p2.txt', 'w', encoding='utf-8')

buffer: List[str] = []
currentLyricId = ''

def bufferMapper(line: str, i: int):
    if(i + 3 < len(buffer)):
        return '\t'.join([currentLyricId,'；'.join(buffer[i:i+4])])
    else:
        return None


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
            buffer = []
        currentLyricId = line[match.start(1):match.end(1)]
    else:
        buffer.append(line.strip())