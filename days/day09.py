from helpers import *

test_data = test_input("""
0 3 6 9 12 15
1 3 6 10 15 21
10 13 16 21 30 45
""")

test_case(1, test_data, 114)
test_case(2, test_data, 2)


def part1_and_2(d: Input, ans: Answers) -> None:
    part1 = 0
    part2 = 0
    for l in d.lines:
        last_stack = []
        first_stack = []
        vals = np.array(l.as_ints, dtype=np.int64)
        last_stack.append(vals[-1])
        first_stack.append(vals[0])
        while True:
            new = vals[1:] - vals[:-1].copy()
            last_stack.append(new[-1])
            first_stack.append(new[0])
            vals = new
            if not np.any(new):
                break

        new_number = sum(last_stack)
        part1 += new_number

        cum = 0
        for i in reversed(first_stack):
            cum = i - cum

        part2 += cum

    ans.part1 = part1
    ans.part2 = part2


run([1, 2], day=9, year=2023, submit=True)
