import numpy

from helpers import *

test_data = test_input(r"""
.|...\....
|.-.\.....
.....|-...
........|.
..........
.........\
..../.\\..
.-.-/..|..
.|....-|.\
..//.|....
""")

test_case(1, test_data, 46)
test_case(2, test_data, 51)

@dataclasses.dataclass
class Beam:
    position: complex
    direction: complex

def trace(starting_position, direction, map):
    beams = [Beam(starting_position, direction)]

    seen = set()
    visited = set()
    visited_map = numpy.zeros(shape=(map.rows, map.columns), dtype=int)
    while beams:
        for beam in list(beams):
            if (beam.position, beam.direction) in seen:
                beams.remove(beam)

            if map[beam.position] == " ":
                beams.remove(beam)
                break

            visited.add(beam.position)
            visited_map[int(beam.position.imag), int(beam.position.real)] = 1
            seen.add((beam.position, beam.direction))

            if map[beam.position] == "\\":
                # change x to y and y to x
                beam.direction = complex(beam.direction.imag, beam.direction.real)

            if map[beam.position] == "/":
                # change x to y and y to x
                beam.direction = complex(-beam.direction.imag, -beam.direction.real)

            if map[beam.position] == "-":
                # vertical
                if beam.direction.real == 0:
                    beams.remove(beam)
                    beams.append(Beam(beam.position, 1))
                    beams.append(Beam(beam.position, -1))

            if map[beam.position] == "|":
                # horizontal
                if beam.direction.imag == 0:
                    beams.remove(beam)
                    beams.append(Beam(beam.position, 1j))
                    beams.append(Beam(beam.position, -1j))

            beam.position += beam.direction

    return len(visited)

def part1_and_2(d: Input, ans: Answers) -> None:
    map = SparseComplexMap(d.lines, default=" ")

    ans.part1 = trace(0, 1, map)

    energized = set()
    for y in range(map.rows):
        energized.add(trace(complex(0, y), 1, map))
        energized.add(trace(complex(map.columns - 1, y), -1, map))

    for x in range(map.rows):
        energized.add(trace(complex(x, 0), 1j, map))
        energized.add(trace(complex(x, map.rows - 1), -1j, map))

    ans.part2 = max(energized)

run([1, 2], day=16, year=2023, submit=True)
