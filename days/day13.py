from helpers import *

test_data = test_input(
    """
    #.##..##.
    ..#.##.#.
    ##......#
    ##......#
    ..#.##.#.
    ..##..##.
    #.#.##.#.
    
    #...##..#
    #....#..#
    ..##..###
    #####.##.
    #####.##.
    ..##..###
    #....#..#
    """
)

test_data2 = test_input(
    """
    ......#
    ###.#..
    ###.##.
    ###.##.
    ###.#..
    .....##
    ##..#..
    ##.#...
    .###.#.
    ##.....
    ..#...#
    #....##
    #....##
    ..#...#
    ##.....
    """
)

test_case(1, test_data, 405)
test_case(1, test_data2, 1200)

test_case(
    1, test_input(
        """
        #.
        ..
        """
    ), 0
)

test_case(
    1, test_input(
        """
        #.#
        """
    ), 0
)

test_case(
    1, test_input(
        """
        #
        .
        #
        """
    ), 0
)

test_case(1, test_input("##"), 1)

test_case(
    1, test_input(
        """
        #
        #
        """
    ), 100
)

test_case(2, test_data, 400)


def v_mirror_position(m):
    """
    Find the location (if any) of a vertically placed mirror in the matrix.
    """
    # iterate over x-coordinates
    mirror_positions = 0
    for x in range(1, m.shape[1]):
        # if the matrix to the right of the x-coordinate is the same as the
        # matrix to the left of the x-coordinate flipped
        map_left = m[:, :x]
        map_right = np.flip(m[:, x:], axis=1)

        # now if left has 4 columns and right has 3 columns, then we want to
        # compare the last 3 columns of left with the first 3 columns of right
        # so we want to compare the last min(left, right) columns of left
        # with the last min(left, right) columns of right

        # if the matrices are the same, then we have a mirror
        part_left = map_left[:, -map_right.shape[1]:]
        part_right = map_right[:, -map_left.shape[1]:]

        if np.array_equal(part_left, part_right):
            # print(m)
            # print("v-mirror", map_left, map_right, part_left, part_right)
            mirror_positions += x

    return mirror_positions


def h_mirror_position(m):
    """
    Find the location (if any) of a horizontally placed mirror in the matrix.
    """
    # iterate over y-coordinates
    mirror_positions = 0
    for y in range(1, m.shape[0]):
        # if the matrix to the bottom of the y-coordinate is the same as the
        # matrix to the top of the y-coordinate flipped
        map_top = m[:y, :]
        map_bottom = np.flip(m[y:, :], axis=0)

        # now if top has 4 rows and bottom has 3 rows, then we want to
        # compare the last 3 rows of top with the first 3 rows of bottom
        # so we want to compare the last min(top, bottom) rows of top with
        # the first min(top, bottom) rows of bottom

        part_top = map_top[-map_bottom.shape[0]:, :]
        part_bottom = map_bottom[-map_top.shape[0]:, :]

        # if the matrices are the same, then we have a mirror
        if np.array_equal(part_top, part_bottom):
            mirror_positions += y

    return mirror_positions


def v_mirror_position_with_smudge(m):
    """
    Find the location (if any) of a vertically placed mirror in the matrix.
    """
    # iterate over x-coordinates
    mirror_positions = 0
    for x in range(1, m.shape[1]):
        # if the matrix to the right of the x-coordinate is the same as the
        # matrix to the left of the x-coordinate flipped
        map_left = m[:, :x]
        map_right = np.flip(m[:, x:], axis=1)

        # now if left has 4 columns and right has 3 columns, then we want to
        # compare the last 3 columns of left with the first 3 columns of right
        # so we want to compare the last min(left, right) columns of left
        # with the last min(left, right) columns of right

        # if the matrices are the same, then we have a mirror
        part_left = map_left[:, -map_right.shape[1]:]
        part_right = map_right[:, -map_left.shape[1]:]

        # now this is a match only if one of the elements *differs* between
        # the two matrices, i.e. the difference between the two matrices is
        # a matrix of all zeros except for one element which is 1 or -1

        # so we can do this by subtracting the two matrices and then checking
        # if the sum of the absolute values of the elements is 1

        diff = part_left - part_right
        print(sum_diff := np.sum(np.abs(diff)))
        if sum_diff == 1:
            mirror_positions += x

    return mirror_positions


def h_mirror_position_with_smudge(m):
    """
    Find the location (if any) of a horizontally placed mirror in the matrix.
    """
    # iterate over y-coordinates
    mirror_positions = 0
    for y in range(1, m.shape[0]):
        # if the matrix to the bottom of the y-coordinate is the same as the
        # matrix to the top of the y-coordinate flipped
        map_top = m[:y, :]
        map_bottom = np.flip(m[y:, :], axis=0)

        # now if top has 4 rows and bottom has 3 rows, then we want to
        # compare the last 3 rows of top with the first 3 rows of bottom
        # so we want to compare the last min(top, bottom) rows of top with
        # the first min(top, bottom) rows of bottom

        part_top = map_top[-map_bottom.shape[0]:, :]
        part_bottom = map_bottom[-map_top.shape[0]:, :]

        # likewise, this is a match only if one of the elements *differs*
        # between the two matrices, i.e. the difference between the two
        # matrices is a matrix of all zeros except for one element which is
        # 1 or -1

        # so we can do this by subtracting the two matrices and then checking
        # if the sum of the absolute values of the elements is 1

        diff = part_top - part_bottom
        if np.sum(np.abs(diff)) == 1:
            mirror_positions += y

    return mirror_positions


def part1(d: Input, ans: Answers) -> None:
    result = 0
    for m in d.paragraphs():
        m = m.replace('#', '1').replace('.', '0')
        m = np.array([list(map(int, row)) for row in m.splitlines()], dtype=int)
        result += v_mirror_position(m)
        result += h_mirror_position(m) * 100

    ans.part1 = result


def part2(d: Input, ans: Answers) -> None:
    result = 0
    for m in d.paragraphs():
        m = m.replace('#', '1').replace('.', '0')
        m = np.array([list(map(int, row)) for row in m.splitlines()], dtype=int)
        result += v_mirror_position_with_smudge(m)
        result += h_mirror_position_with_smudge(m) * 100

    ans.part2 = result


run([1, 2], day=13, year=2023, submit=True)
