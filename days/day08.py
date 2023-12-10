from helpers import *

test_data1 = test_input("""
RL

AAA = (BBB, CCC)
BBB = (DDD, EEE)
CCC = (ZZZ, GGG)
DDD = (DDD, DDD)
EEE = (EEE, EEE)
GGG = (GGG, GGG)
ZZZ = (ZZZ, ZZZ)
""")

test_data2 = test_input("""
LLR

AAA = (BBB, BBB)
BBB = (AAA, ZZZ)
ZZZ = (ZZZ, ZZZ)
""")


test_data3 = test_input("""
LR

11A = (11B, XXX)
11B = (XXX, 11Z)
11Z = (11B, XXX)
22A = (22B, XXX)
22B = (22C, 22C)
22C = (22Z, 22Z)
22Z = (22B, 22B)
XXX = (XXX, XXX)
""")

test_case(1, test_data1, 2)
test_case(1, test_data2, 6)
test_case(2, test_data3, 6)

def parse(d) -> tuple[str, dict[str, tuple[str, str]]]:
    instructions, tree = d.paragraphs()

    tree_nodes = {}
    for node, l, r in tree.parsed_lines('<> = (<>, <>)'):
        tree_nodes[node] = (l, r)

    return instructions, tree_nodes


def part1(d: Input, ans: Answers) -> None:
    instructions, tree_nodes = parse(d)

    current_node = 'AAA'
    for n, i in enumerate(cycle(instructions), 1):
        if i == 'L':
            current_node = tree_nodes[current_node][0]
        elif i == 'R':
            current_node = tree_nodes[current_node][1]
        else:
            assert False

        if current_node == 'ZZZ':
            ans.part1 = n
            break


def part2(d: Input, ans: Answers) -> None:
    instructions, tree_nodes = parse(d)

    current_nodes = [i for i in tree_nodes if i.endswith('A')]

    cycle_starts = [{} for _ in current_nodes]
    seen_at = [{} for _ in current_nodes]
    cycle_recorded_for = set()

    known_cycles_for_ghosts = {}
    iters_remaining = None
    for n, (instr_pos, instr) in enumerate(cycle(enumerate(instructions)), 1):
        for idx, c in enumerate(current_nodes):
            if instr == 'L':
                new = tree_nodes[c][0]
            elif instr == 'R':
                new = tree_nodes[c][1]
            else:
                assert False

            current_nodes[idx] = new
            if new.endswith('Z'):
                seen_at_for = seen_at[idx]
                position = (instr_pos, new)
                if position in seen_at_for:
                    if position not in cycle_recorded_for:
                        cycle_recorded_for.add(position)
                        cycle_starts[idx][seen_at_for[position]] = n - seen_at_for[position]
                        known_cycles_for_ghosts[idx] = n - seen_at_for[position]
                else:
                    seen_at_for[position] = n

        if iters_remaining is not None:
            if iters_remaining < 0:
                break

            iters_remaining -= 1
        else:
            if len(known_cycles_for_ghosts) == len(current_nodes):
                iters_remaining = max(known_cycles_for_ghosts.values())

        # bail out early if we've seen all cycles
        if all(c.endswith('Z') for c in current_nodes):
            ans.part2 = n
            return

    cycles = {}
    for i in cycle_starts:
        cycles.update(i)

    solution = crt(mods=[(length, start % length) for (start, length) in list(cycles.items())])
    ans.part2 = solution.cycle

run([1, 2], day=8, year=2023, submit=True)
