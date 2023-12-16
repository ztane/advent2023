from helpers import *

test_data = test_input("""
seeds: 79 14 55 13

seed-to-soil map:
50 98 2
52 50 48

soil-to-fertilizer map:
0 15 37
37 52 2
39 0 15

fertilizer-to-water map:
49 53 8
0 11 42
42 0 7
57 7 4

water-to-light map:
88 18 7
18 25 70

light-to-temperature map:
45 77 23
81 45 19
68 64 13

temperature-to-humidity map:
0 69 1
1 0 69

humidity-to-location map:
60 56 37
56 93 4
""")

test_case(1, test_data, 35)
test_case(2, test_data, 46)




def find_lowest(starting_tree: IntervalTree, instructions: list[Input]) -> int:
    t = starting_tree
    for i in instructions:
        new_intervals = []
        for j in i.lines[1:]:
            dest, source, amount = j.as_ints
            t.slice(source)
            t.slice(source + amount)

            for ival in t[source:source + amount]:
                t.remove(ival)
                delta = dest - source
                new_intervals.append(Interval(ival.begin + delta, ival.end + delta))

        for ival in new_intervals:
            t.add(ival)

    return t.begin()


def part1_and_2(d: Input, ans: Answers) -> None:
    first, *rest = d.paragraphs()

    seed_range_info = list(first.as_ints)

    p1 = IntervalTree()
    for start in seed_range_info:
        p1.add(Interval(start, start + 1))

    p2 = IntervalTree()
    for start, length in zip(seed_range_info[::2], seed_range_info[1::2]):
        p2.add(Interval(start, start + length))

    ans.part1 = find_lowest(p1, rest)
    ans.part2 = find_lowest(p2, rest)


run([1, 2], day=5, year=2023, submit=True)
