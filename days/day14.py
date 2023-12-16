import itertools
import hashlib
import numpy

from helpers import *

test_data = test_input("""
O....#....
O.OO#....#
.....##...
OO.#O....O
.O.....O#.
O.#..O.#.#
..O..#O..O
.......O..
#....###..
#OO..#....
""")

test_case(1, test_data, 136)
test_case(2, test_data, 64)


def roll_north(m):
    rows = m.shape[0]
    columns = m.shape[1]

    for x, y in itertools.product(range(columns), range(rows)):
        if m[y, x] == "O":
            # move the rock north:
            new_y = y
            for i in range(y - 1, -1, -1):
                if m[i, x] != ".":
                    break

                new_y = i

            m[y, x] = "."
            m[new_y, x] = "O"

def calculate_load(m):
    rows = m.shape[0]
    columns = m.shape[1]

    total = 0
    for x, y in itertools.product(range(columns), range(rows)):
        if m[y, x] == "O":
            load = rows - y
            total += load

    return total

def part1(d: Input, ans: Answers) -> None:
    m = d.char_matrix
    roll_north(m)
    ans.part1 = calculate_load(m)

def cycle(m):
    for i in range(4):
        roll_north(m)
        m = numpy.rot90(m, k=3)

    return m

def part2(d: Input, ans: Answers) -> None:
    m = d.char_matrix
    roll_north(m)
    ans.part1 = calculate_load(m)

    total_run = 1000000000

    previous = {}
    i = 0
    cycle_length = 0
    for i in range(total_run):
        m = cycle(m)
        h = hash_array(m)
        if h in previous:
            cycle_length = i - previous[h]
            print(f"Found cycle of length {cycle_length} at {i}")
            break

        previous[h] = i

    # now we have the cycle length, we can just do the remainder of the
    # division to get the final answer

    remaining_iterations = total_run - i - 1
    remaining_iterations %= cycle_length

    for i in range(remaining_iterations):
        m = cycle(m)

    ans.part2 = calculate_load(m)


run([1, 2], day=14, year=2023, submit=True)
