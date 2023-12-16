import itertools

from helpers import *

test_data = test_input(
    """
    ...#......
    .......#..
    #.........
    ..........
    ......#...
    .#........
    .........#
    ..........
    .......#..
    #...#.....
    """
)

test_case(1, test_data, 374)


def part1_and_2(d: Input, ans: Answers) -> None:
    star_map = SparseMap(d.lines, default=".")
    galaxies = []
    galaxy_rows = set()
    galaxy_columns = set()
    for x, y in star_map:
        if star_map[x, y] == "#":
            galaxies.append((x, y))
            galaxy_rows.add(y)
            galaxy_columns.add(x)

    x_expansions = []
    current_expansion = 0
    for x in star_map.columns:
        if x not in galaxy_columns:
            current_expansion += 1

        x_expansions.append(current_expansion)

    y_expansions = []
    current_expansion = 0
    for y in star_map.rows:
        if y not in galaxy_rows:
            current_expansion += 1

        y_expansions.append(current_expansion)

    answers = []
    for expansion_multiplier in [1, 1000000 - 1]:
        expanded = [
            (
                x + expansion_multiplier * x_expansions[x],
                y + expansion_multiplier * y_expansions[y],
            )
            for x, y in galaxies
        ]

        total_distances = 0

        for (g1, g2) in itertools.combinations(expanded, 2):
            total_distances += abs(g1[0] - g2[0]) + abs(g1[1] - g2[1])

        answers.append(total_distances)

    ans.part1, ans.part2 = answers



run([1, 2], day=11, year=2023, submit=True)
