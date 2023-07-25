from typing import List
import numpy as np
from chess import pgn, Board, Color


def board_to_piece_squares(board: Board, color: Color) -> np.ndarray:
    piece_squares = np.zeros(64, dtype=int)
    for square in range(64):
        piece = board.piece_at(square)
        # Check if piece exists on square
        if piece is not None:
            mapped_piece = piece.piece_type
            # Negate piece_type for black pieces since piece_type is positive for black and white
            if not piece.color:
                mapped_piece *= -1
            square_index = square if color else 63 - square
            piece_squares[square_index] = mapped_piece
    if not color:
        piece_squares *= -1
    return piece_squares


def process_game(game: pgn.Game, out_file):
    state = game.next()
    while state:
        evaluation = state.eval()
        # In case of only partly analysed games
        if evaluation is None:
            return

        score = -evaluation.relative.score(mate_score=10000) / 100
        score = np.tanh(score / 10)
        piece_squares = board_to_piece_squares(state.board(), state.turn())

        # Positive score --> good evaluation for side that performed the last move
        # Side that performed last move has positive values in piece_squares array
        # print(score)
        # print(state.board())
        # print(piece_squares)
        out_file.write(','.join(map(str, piece_squares)) + ',' + str(score) + '\n')

        state = state.next()


def preprocess(max=10000) -> None:
    with open('dataset/processed.csv', 'w') as out_file:

        with open("dataset/lichess_db_standard_rates_2023-05_000.pgn", 'r') as in_file:
            game_counter = 0
            while game_counter < max:
                game = pgn.read_game(in_file)
                if game is None:
                    continue

                # Check if game is evaluated
                state = game.next()
                if state is None:
                    continue
                if state.eval() is None:
                    continue

                process_game(game, out_file)
                game_counter += 1

preprocess()