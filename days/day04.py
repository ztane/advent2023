from helpers import *

test_data = test_input("""
Card 1: 41 48 83 86 17 | 83 86  6 31 17  9 48 53
Card 2: 13 32 20 16 61 | 61 30 68 82 17 32 24 19
Card 3:  1 21 53 59 44 | 69 82 63 72 16 21 14  1
Card 4: 41 92 73 84 69 | 59 84 76 51 58  5 54 83
Card 5: 87 83 26 28 32 | 88 30 70 12 93 22 82 36
Card 6: 31 18 13 56 72 | 74 77 10 23 35 67 36 11
""")

test_case(1, test_data, 13)
test_case(2, test_data, 30)


def part1(d: Input, ans: Answers) -> None:
    ans.part1 = 0
    for cardno, drawn, own in d.parsed_lines("Card <int>: <> | <>"):
        drawn_nos = Input(drawn).as_ints
        own_nos = Input(own).as_ints

        total = len(set(own_nos).intersection(drawn_nos))
        if total > 0:
            ans.part1 += 2 ** (total - 1)


def part2(d: Input, ans: Answers) -> None:
    ans.part2 = 0

    n_cards = defaultdict(lambda: 1)
    for cardno, drawn, own in d.parsed_lines("Card <int>: <> | <>"):
        drawn_nos = drawn.as_ints
        own_nos = own.as_ints
        new_cards = len(set(own_nos).intersection(drawn_nos))

        for i in range(new_cards):
            n_cards[cardno + i + 1] += n_cards[cardno]

        ans.part2 += n_cards[cardno]


run([1, 2], day=4, year=2023, submit=True)
