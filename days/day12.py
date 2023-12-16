from helpers import *

test_data = test_input("""
#.#.### 1,1,3
.#...#....###. 1,1,3
.#.###.#.###### 1,3,1,6
####.#...#... 4,1,1
#....######..#####. 1,6,5
.###.##....# 3,2,1
""")

test_data2 = test_input("""
???.### 1,1,3
.??..??...?##. 1,1,3
?#?#?#?#?#?#?#? 1,3,1,6
????.#...#... 4,1,1
????.######..#####. 1,6,5
?###???????? 3,2,1
""")

test_case(1, test_data2, 21)
test_case(2, test_data2, 525152)


@lru_cache(maxsize=None)
def calculate_combinations(records, lengths):
    loc = 0
    total_combinations = 0
    if lengths and not records.strip('.'):
        return 0

    if not lengths:
        if '#' in records:
            return 0

        return 1

    while loc < len(records):
        length = lengths[0]
        if loc + length > len(records):
            break

        c = records[loc]
        if c in '#?':
            for i in range(length):
                if records[loc + i] not in '#?':
                    break
            else:
                if loc + length == len(records) or records[loc + length] in ('?', '.'):
                    total_combinations += calculate_combinations(records[loc + length + 1:], lengths[1:])

            if c == '#':
                break

        loc += 1

    return total_combinations


assert calculate_combinations('?###????????.', (3,2,1)) == 10


def part1(d: Input, ans: Answers) -> None:
    total = 0
    for i in d.lines:
        records, coords = i.split()
        coords = tuple(coords.as_ints)

        result = calculate_combinations(records + '.', coords)
        calculate_combinations.cache_clear()
        total += result
        print(records, coords, result)

    ans.part1 = total



def part2(d: Input, ans: Answers) -> None:
    total = 0
    for i in d.lines:
        records, coords = i.split()
        coords = coords.as_ints
        records = '?'.join([records] * 5)
        coords = coords * 5

        print(records, coords)
        result = calculate_combinations(records + '.', coords)
        total += result

    ans.part2 = total


run([1,2], day=12, year=2023, submit=True)
