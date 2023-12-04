from helpers import *

test_data1 = test_input(
    """
    1abc2
    pqr3stu8vwx
    a1b2c3d4e5f
    treb7uchet
    """
)

test_case(1, test_data1, 142)

test_data2 = test_input(
    """
    two1nine
    eightwothree
    abcone2threexyz
    xtwone3four
    4nineeightseven2
    zoneight234
    7pqrstsixteen
    """
)

test_case(2, test_data2, 281)


def part1(d: Input, ans: Answers) -> None:
    ans.part1 = 0
    for i in d.lines:
        digits = [int(c) for c in i if c.isdigit()]
        ans.part1 += digits[0] * 10 + digits[-1]


digits_part_2 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

digits_as_words_reversed = [i[::-1] for i in digits_part_2]


def part2(d: Input, ans: Answers) -> None:
    ans.part2 = 0
    pattern = re.compile("|".join(digits_part_2))
    reverse_pattern = re.compile("|".join(digits_as_words_reversed))

    for i in d.lines:
        first_digit = pattern.search(i)
        last_digit = reverse_pattern.search(i[::-1])

        ans.part2 += (
            digits_part_2.index(first_digit.group()) % 10 * 10
            + digits_part_2.index(last_digit.group()[::-1]) % 10
        )


run([1, 2], day=1, year=2023, submit=True)
