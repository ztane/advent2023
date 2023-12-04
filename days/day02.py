from helpers import *

test_data = test_input("""
Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green
Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue
Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red
Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red
Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green
""")

test_case(1, test_data, 8)
test_case(2, test_data, 2286)


def part1(d: Input, ans: Answers) -> None:
    max_counts = {"red": 12, "green": 13, "blue": 14}
    ans.part1 = 0
    for game_id, sets in d.parsed_lines("Game <int>: <str>"):
        failed = False
        for i in sets.split("; "):
            for j in i.split(", "):
                count, colour = j.parsed("<int> <>")
                if count > max_counts[colour]:
                    failed = True
                    break

        if not failed:
            ans.part1 += game_id


def part2(d: Input, ans: Answers) -> None:
    ans.part2 = 0
    for game_id, sets in d.parsed_lines("Game <int>: <str>"):
        counts = defaultdict(int)
        for i in sets.split("; "):
            for j in i.split(", "):
                count, colour = j.parsed("<int> <>")
                if count > counts[colour]:
                    counts[colour] = count

        ans.part2 += prod(counts.values())

run([1, 2], day=2, year=2023, submit=True)
