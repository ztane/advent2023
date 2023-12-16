from helpers import *

test_data = test_input("""
32T3K 765
T55J5 684
KK677 28
KTJJT 220
QQQJA 483
""")

test_case(1, test_data, 6440)
test_case(2, test_data, 5905)

card_to_value: dict[str, int] = dict((v, k) for (k, v) in enumerate('23456789TJQKA', 2))

def to_hand(s: str, jokers=False) -> tuple[int, int, int, int, int]:
    rv = []
    for c in s:
        value = card_to_value[c]
        if jokers and value == 11:
            value = 0
        rv.append(value)

    assert(len(rv) == 5)
    return cast(tuple[int, int, int, int, int], tuple(rv))


def get_hand_rank(hand: tuple[int, int, int, int, int]) -> int:
    set_vals = tuple(sorted(Counter(hand).values()))

    if set_vals == (5,):
        # five of a kind
        return 7

    if set_vals == (1, 4):
        # four of a kind
        return 6

    if set_vals == (2, 3):
        # full house
        return 5

    if set_vals == (1, 1, 3):
        # three of a kind
        return 4

    if set_vals == (1, 2, 2):
        # two pair
        return 3

    if set_vals == (1, 1, 1, 2):
        # pair
        return 2

    if set_vals == (1, 1, 1, 1, 1):
        # high card
        return 1

    assert False, f"Invalid hand: {hand}"


def get_joker_hand_rank(hand: tuple[int, int, int, int, int]) -> int:
    joker_positions = [i for i, v in enumerate(hand) if v == 0]

    if len(joker_positions) == 0:
        return get_hand_rank(hand)

    max_rank = 1
    hand_mut = list(hand)
    for j in range(2, 15):
        for i in joker_positions:
            hand_mut[i] = j

        max_rank = max(max_rank, get_hand_rank(cast(tuple[int, int, int, int, int], tuple(hand_mut))))

    return max_rank

def part1(d: Input, ans: Answers) -> None:
    total_hands = []
    all_hands = set()
    for hand_s, bet in d.parsed_lines('<> <int>'):
        hand = to_hand(hand_s)

        hand_rank = get_hand_rank(hand)
        all_hands.add(hand)
        total_hands.append((hand_rank, hand, bet))

    total_hands.sort()
    totals = 0
    for idx, (hand_rank, hand, bet) in enumerate(total_hands, 1):
        print(idx, hand_rank, hand, bet)
        totals += bet * idx

    ans.part1 = totals

def part2(d: Input, ans: Answers) -> None:
    total_hands = []
    all_hands = set()
    for hand_s, bet in d.parsed_lines('<> <int>'):
        hand = to_hand(hand_s, jokers=True)

        hand_rank = get_joker_hand_rank(hand)
        all_hands.add(hand)
        total_hands.append((hand_rank, hand, bet))

    total_hands.sort()
    totals = 0
    for idx, (hand_rank, hand, bet) in enumerate(total_hands, 1):
        totals += bet * idx

    ans.part2 = totals


run([1, 2], day=7, year=2023, submit=True)
