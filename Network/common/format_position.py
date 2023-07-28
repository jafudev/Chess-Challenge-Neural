from typing import List

NUMBER_TO_PIECE_MAPPING = {
    0: "-",
    -1: "\u2659",
    -2: "\u2658",
    -3: "\u2657",
    -4: "\u2656",
    -5: "\u2655",
    -6: "\u2654",
    1: "\u265F",
    2: "\u265E",
    3: "\u265D",
    4: "\u265C",
    5: "\u265B",
    6: "\u265A"
}


def format_position(position: List):
    for i in range(8):
        for j in range(8):
            piece = position[64 - (i + 1) * 8 + j]
            print(f"{NUMBER_TO_PIECE_MAPPING[piece]}\t", end="")
        print()


format_position(
    [4,2,3,6,5,3,2,4,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1,-1,0,-1,-1,-1,-1,-4,-2,-3,-6,-5,-3,-2,-4])
print()

