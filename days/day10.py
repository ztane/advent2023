from helpers import *

test_data = test_input("""
..F7.
.FJ|.
SJ.L7
|F--J
LJ...
""")

test_data2 = test_input("""
.....
.S-7.
.|.|.
.L-J.
.....
""")


test_data3 = test_input("""
.F----7F7F7F7F-7....
.|F--7||||||||FJ....
.||.FJ||||||||L7....
FJL7L7LJLJ||LJ.L-7..
L--J.L7...LJS7F-7L7.
....F-J..F7FJ|L7L7L7
....L7.F7||L7|.L7L7|
.....|FJLJ|FJ|F7|.LJ
....FJL-7.||.||||...
....L---J.LJ.LJLJ...
""")

test_data4 = test_input("""
FF7FSF7F7F7F7F7F---7
L|LJ||||||||||||F--J
FL-7LJLJ||||||LJL-77
F--JF--7||LJLJ7F7FJ-
L---JF-JLJ.||-FJLJJ7
|F|F-JF---7F7-L7L|7|
|FFJF7L7F-JF7|JL---7
7-L-JL7||F7|L7F-7F7|
L.L7LFJ|||||FJL7||LJ
L7JLJL-JLJLJL--JLJ.L
""")

test_case(1, test_data, 8)
test_case(1, test_data2, 4)
test_case(2, test_data4, 10)
test_case(2, test_data3, 8)

replacements = {
    "S": "╬",
    "F": "╔",
    "J": "╝",
    "7": "╗",
    "L": "╚",
    "|": "║",
    "-": "═",
}

connections = defaultdict(list, {
    # (x, y)
    # south and east
    "╔": [(1, 0), (0, 1)],
    # north and west
    "╝": [(-1, 0), (0, -1)],

    # south and west
    "╗": [(-1, 0), (0, 1)],
    # north and east
    "╚": [(1, 0), (0, -1)],

    # south and north
    "║": [(0, 1), (0, -1)],
    # east and west
    "═": [(1, 0), (-1, 0)],

    # all connections
    "╬": [(1, 0), (-1, 0), (0, 1), (0, -1)],
})

translator = better_translator(replacements)


def part1(d: Input, ans: Answers) -> None:
    d = Input(translator(d))
    sparse_map = SparseMap(d.lines, default=".")
    distances = {}

    start = None
    for x, y in sparse_map:
        if sparse_map[x, y] == "╬":
            start = (x, y)
            break

    assert start is not None

    queue = []
    heapq.heappush(queue, (0, start))

    while queue:
        dist, (x, y) = heapq.heappop(queue)
        if (x, y) in distances:
            continue

        distances[x, y] = dist

        map_char = sparse_map[x, y]
        for dx, dy in connections[map_char]:
            new_x = x + dx
            new_y = y + dy
            character = sparse_map[new_x, new_y]

            if not connections[character]:
                continue

            for i in connections[character]:
                if i == (-dx, -dy):
                    heapq.heappush(queue, (dist + 1, (new_x, new_y)))
                    break

    ans.part1 = max(distances.values())


def part2(d: Input, ans: Answers) -> None:
    d = Input(translator(d))
    sparse_map = SparseMap(d.lines, default="O")
    distances = {}

    start = None
    for x, y in sparse_map:
        if sparse_map[x, y] == "╬":
            start = (x, y)
            break

    assert start is not None

    queue = []
    heapq.heappush(queue, (0, start))

    while queue:
        dist, (x, y) = heapq.heappop(queue)
        if (x, y) in distances:
            continue

        distances[x, y] = dist

        map_char = sparse_map[x, y]
        for dx, dy in connections[map_char]:
            new_x = x + dx
            new_y = y + dy
            character = sparse_map[new_x, new_y]

            if not connections[character]:
                continue

            for i in connections[character]:
                if i == (-dx, -dy):
                    heapq.heappush(queue, (dist + 1, (new_x, new_y)))
                    break

    for x, y in sparse_map:
        if (x, y) not in distances:
            sparse_map[x, y] = "."

    # replace start with correct loop-closing character:
    start_x, start_y = start
    my_conns = []
    for dx, dy in connections[sparse_map[start_x, start_y]]:
        char = sparse_map[start_x + dx, start_y + dy]
        if (-dx, -dy) in connections[char]:
            my_conns.append((dx, dy))

    for c, i in connections.items():
        if set(my_conns) == set(i):
            sparse_map[start_x, start_y] = c
            break

    total_in = 0
    for y in sparse_map.rows:
        inside = False
        from_down = False
        from_up = False
        for x in sparse_map.columns:
            if inside and sparse_map[x, y] == ".":
                total_in += 1
            if sparse_map[x, y] == "║":
                inside = not inside
            if sparse_map[x, y] == "╔":
                from_down = True
            if sparse_map[x, y] == "╗":
                if from_up:
                    inside = not inside
                from_up = from_down = False
            if sparse_map[x, y] == "╝":
                if from_down:
                    inside = not inside
                from_up = from_down = False
            if sparse_map[x, y] == "╚":
                from_up = True

    ans.part2 = total_in


run([1, 2], day=10, year=2023, submit=True)
