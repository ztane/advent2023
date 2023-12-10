from helpers import *

test_data = test_input("""
Time:      7  15   30
Distance:  9  40  200
""")

test_case(1, test_data, 288)
test_case(2, test_data, 71503)


def part1(d: Input, ans: Answers) -> None:
    times = d.lines[0].as_ints
    distances = d.lines[1].as_ints

    ans.part1 = 1
    for t, d in zip(times, distances):
        print(t, d)

        possible = 0
        for hold_time in range(t):
            speed = hold_time
            remaining = t - hold_time
            distance = speed * remaining

            if distance > d:
                possible += 1

        ans.part1 *= possible

def part2(d: Input, ans: Answers) -> None:
    time = int(''.join(str(i) for i in d.lines[0].as_ints))
    distance = int(''.join(str(i) for i in d.lines[1].as_ints))

    # hold_time * (t - hold_time) > distance
    # hold_time * t - hold_time^2 > distance
    # hold_time^2 - hold_time * t + distance < 0
    # hold_time = (t +- sqrt(t^2 - 4 * distance)) / 2

    hold_time_min = ceil((time - sqrt(time**2 - 4 * distance)) / 2)
    hold_time_max = floor((time + sqrt(time**2 - 4 * distance)) / 2)
    ans.part2 = hold_time_max - hold_time_min + 1

run([1, 2], day=6, year=2023, submit=True)
