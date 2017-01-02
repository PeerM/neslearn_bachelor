import itertools
from pprint import pprint
order = ["right", "left", "down", "up", "start", "select", "B", "A"]
base_offset = 3

def parse_fm2(file):
    for line in file.readlines():
        if line[0] != "|":
            continue
        yield {order[i]:line[base_offset+i]!="." for i in range(8)}

if __name__ == '__main__':
    with open("../bin_deps/fceux/movies/happylee-supermariobros,warped.fm2") as file:
        # print(list(itertools.islice(parse_fm2(file),40)))
        for combi in parse_fm2(file):
            if any(combi.values()):
                print(combi)
       #  pprint(list(parse_fm2(file))[560])


