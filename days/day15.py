from helpers import *

test_data = test_input("""
rn=1,cm-,qp=3,cm=2,qp-,pc=4,ot=9,ab=5,pc-,pc=6,ot=7
""")

test_case(1, test_data, 1320)
test_case(2, test_data, 145)


def HASH(s: str) -> int:
    c = 0
    for i in s:
        i = ord(i)
        c += i
        c *= 17
        c %= 256

    return c

assert HASH('HASH') == 52

def part1(d: Input, ans: Answers) -> None:
    parts = d.split(",")
    ans.part1 = sum(HASH(i) for i in parts)


def part2(d: Input, ans: Answers) -> None:
    boxes: dict[int, list[tuple[str, int]]] = defaultdict(list[tuple[str, int]])

    for i in d.split(","):
        if i.endswith("-"):
            lens = i[:-1]
            box_number = HASH(lens)

            for idx, box_lens in enumerate(boxes[box_number]):
                if box_lens[0] == lens:
                    boxes[box_number].pop(idx)
                    break

        else:
            label, fl = i.parsed("<>=<int>")
            box_number = HASH(label)
            box = boxes[box_number]

            for idx, (blens, bfl) in enumerate(box):
                if blens == label:
                    box[idx] = (blens, fl)
                    break
            else:
                box.append((label, fl))

    part2 = 0
    for box_no, contents in boxes.items():
        for slot, (label, fl) in enumerate(contents, 1):
            part2 += (box_no + 1) * slot * fl

    ans.part2 = part2

run([1, 2], day=15, year=2023, submit=True)
