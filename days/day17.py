from helpers import *

test_data = test_input(
    """
2413432311323
3215453535623
3255245654254
3446585845452
4546657867536
1438598798454
4457876987766
3637877979653
4654967986887
4564679986453
1224686865563
2546548887735
4322674655533
"""
)

test_data2 = test_input(
    """
111111111111
999999999991
999999999991
999999999991
999999999991
"""
)

test_case(1, test_data, 102)
test_case(2, test_data, 94)
test_case(2, test_data2, 71)


def part1_and_2(d: Input, ans: Answers) -> None:
    out = 2**32
    d = SparseComplexMap(d.lines, default=out, convert=int)

    t = complex(d.columns - 1, d.rows - 1)

    heuristics_map = SparseComplexMap(default=out)
    # fill the distance heuristics map using a breadth first search
    queue = []
    heapq.heappush(queue, (d[t], t.real, t.imag))

    while queue:
        dist, pos_real, pos_imag = heapq.heappop(queue)
        pos = complex(pos_real, pos_imag)
        if dist >= heuristics_map[pos]:
            continue

        heuristics_map[pos] = dist
        for direction in (1, -1, 1j, -1j):
            new_pos = pos + direction
            if d[new_pos] == out:
                continue

            new_dist = dist + d[new_pos]
            if heuristics_map[new_pos] < new_dist:
                continue

            if d[new_pos] != out:
                heapq.heappush(queue, (new_dist, new_pos.real, new_pos.imag))

    def neighbourhood_p1(p):
        pos, direction, steps = p

        if direction == 0:
            possible = [(1, 1, 1), (1j, 1j, 1)]
        else:
            possible = []
            if steps < 3:
                possible = [(pos + direction, direction, steps + 1)]

            new_dir = direction * 1j
            possible.append((pos + new_dir, new_dir, 1))
            new_dir = direction * -1j
            possible.append((pos + new_dir, new_dir, 1))

        for i in possible:
            if d[i[0]] != out:
                yield d[i[0]], i

    def neighbourhood_p2(p):
        pos, direction, steps = p

        if direction == 0:
            possible = [(1, 1, 1), (1j, 1j, 1)]
        else:
            straight = (pos + direction, direction, steps + 1)
            left = (pos + direction * 1j, direction * 1j, 1)
            right = (pos + direction * -1j, direction * -1j, 1)

            if steps < 4:
                possible = [straight]
            elif steps < 10:
                possible = [straight, left, right]
            else:
                possible = [left, right]

        for i in possible:
            if d[i[0]] != out:
                yield d[i[0]], i

    part1 = a_star_solve(
        (0, 0, 0),
        target=(t, 0, 0),
        heuristic=lambda a, b: heuristics_map[a[0]],
        neighbours=neighbourhood_p1,
        is_target=lambda p: p[0] == t,
    )

    ans.part1 = part1[0]

    part2 = a_star_solve(
        (0, 0, 0),
        target=(t, 0, 0),
        heuristic=lambda a, b: heuristics_map[a[0]],
        neighbours=neighbourhood_p2,
        is_target=lambda p: p[0] == t and p[2] >= 4,
    )

    ans.part2 = part2[0]


run([1, 2], day=17, year=2023, submit=True)
