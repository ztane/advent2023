from helpers import *

test_data = test_input("""
467..114..
...*......
..35..633.
......#...
617*......
.....+.58.
..592.....
......755.
...$.*....
.664.598..
""")

test_case(1, test_data, 4361)
test_case(2, test_data, 467835)


def part1(d: Input, ans: Answers) -> None:
    m = SparseMap(d.lines, default='.')
    max_x = m.columns
    ans.part1 = 0
    not_symbols = set(".0123456789")
    symbols = set()

    for y in range(m.rows):
        x = 0
        run_start = None
        is_valid = False

        while x <= max_x:
            if m[x, y].isdigit():
                if run_start is None:
                    run_start = x
                    is_valid = False

                if not is_valid:
                    for i in neighbourhood_8(x, y):
                        if m[i] not in not_symbols:
                            is_valid = True
                            symbols.add(m[i])
                            break

            else:
                if run_start is not None:
                    if is_valid:
                        digits = ''
                        for i in range(run_start, x):
                            digits += m[i, y]
                        ans.part1 += int(digits)

                    run_start = None
                    is_valid = False

            x += 1

def part2(d: Input, ans: Answers) -> None:
    m = SparseMap(d.lines, default='.')
    max_x = m.columns
    ans.part2 = 0
    not_symbols = set(".0123456789")
    symbols = set()

    starred_runs = defaultdict(set)
    run_numbers = {}


    for y in range(m.rows):
        x = 0
        run_start = None
        is_geared = False

        while x <= max_x:
            if m[x, y].isdigit():
                if run_start is None:
                    run_start = x
                    is_geared = False

                for i in neighbourhood_8(x, y):
                    if m[i] == '*':
                        starred_runs[i].add((y, run_start))
                        is_geared = True

            else:
                if run_start is not None:
                    if is_geared:
                        digits = ''
                        for i in range(run_start, x):
                            digits += m[i, y]

                        run_numbers[y, run_start] = int(digits)

                run_start = None
                is_geared = False

            x += 1

    for k, l in starred_runs.items():
        # only a star with two part numbers is a gear
        if len(l) == 2:
            gear_ratio = run_numbers[l.pop()] * run_numbers[l.pop()]

            ans.part2 += gear_ratio

run([1, 2], day=3, year=2023, submit=True)
