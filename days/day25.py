from helpers import *

test_data = Input(
    """\
5764801
17807724
"""
)

test_case(1, test_data, 14897079)


d = get_aoc_data(day=25)


def transform(subject, loop_number, *, pow=builtin_pow):
    return pow(subject, loop_number, 20201227)


def part1(d: Input, ans: Answers):
    card_public_key, door_public_key = d.extract_ints

    # slower routine, no need to do modpows for recovering
    # key secret, but this shows symmetry here...
    for card_loop_number in count(0):
        if transform(7, card_loop_number) == card_public_key:
            break

    for door_loop_number in count(0):
        if transform(7, door_loop_number) == door_public_key:
            break

    card_enc_key = transform(card_public_key, door_loop_number)
    door_enc_key = transform(door_public_key, card_loop_number)

    assert card_enc_key == door_enc_key
    ans.part1 = card_enc_key


run([1], day=25, year=2020)
