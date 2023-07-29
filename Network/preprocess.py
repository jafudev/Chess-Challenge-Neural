import numpy as np
from common.globals import PROCESSED_DATA_CSV_PATH, LICHESS_DATASET_PATH
from chess import pgn, Board, Color


def board_to_piece_squares(board: Board, color: Color) -> np.ndarray:
    piece_squares = np.zeros(6 * 64, dtype=int)
    for square in range(64):
        piece = board.piece_at(square)
        # Check if piece exists on square
        if piece is not None:
            piece_sign = 1
            # Negate piece_type for black pieces since piece_type is positive for black and white
            if not piece.color:
                piece_sign *= -1
            square_index = square if not color else 63 - square
            piece_squares[square_index + 64 * (piece.piece_type - 1)] = piece_sign
    if color:
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


def preprocess(max_games=1000000) -> None:
    with open(PROCESSED_DATA_CSV_PATH, 'w') as out_file:

        with open(LICHESS_DATASET_PATH, 'r') as in_file:
            game_counter = 0
            while game_counter < max_games:
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