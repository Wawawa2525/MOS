import math
import random
import copy
from concurrent.futures import ThreadPoolExecutor

BLACK = 1
WHITE = 2

board = [
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
        [0,0,1,2,0,0],
        [0,0,2,1,0,0],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
]

def can_place_x_y(board, stone, x, y):
    """
    石を置けるかどうかを調べる関数。
    board: 2次元配列のオセロボード
    x, y: 石を置きたい座標 (0-indexed)
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    return: 置けるなら True, 置けないなら False
    """
    if board[y][x] != 0:
        return False  # 既に石がある場合は置けない

    opponent = 3 - stone  # 相手の石 (1なら2、2なら1)
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        found_opponent = False

        while 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == opponent:
            nx += dx
            ny += dy
            found_opponent = True

        if found_opponent and 0 <= nx < len(board[0]) and 0 <= ny < len(board) and board[ny][nx] == stone:
            return True  # 石を置ける条件を満たす

    return False

def can_place(board, stone):
    """
    石を置ける場所を調べる関数。
    board: 2次元配列のオセロボード
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    """
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                return True
    return False

def random_place(board, stone):
    """
    石をランダムに置く関数。
    board: 2次元配列のオセロボード
    stone: 現在のプレイヤーの石 (1: 黒, 2: 白)
    """
    while True:
        x = random.randint(0, len(board[0]) - 1)
        y = random.randint(0, len(board) - 1)
        if can_place_x_y(board, stone, x, y):
            return x, y


# 評価表
EARLY_EVAL_TABLE = [
    [200, -100, 50, 50, -100, 200],
    [-100, -200, -10, -10, -200, -100],
    [50, -10, 0, 0, -10, 50],
    [50, -10, 0, 0, -10, 50],
    [-100, -200, -10, -10, -200, -100],
    [200, -100, 50, 50, -100, 200],
]

MID_EVAL_TABLE = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50, 0, 0, -50, -20],
    [10, 0, 5, 5, 0, 10],
    [10, 0, 5, 5, 0, 10],
    [-20, -50, 0, 0, -50, -20],
    [100, -20, 10, 10, -20, 100],
]

LATE_EVAL_TABLE = [
    [100, -20, 10, 10, -20, 100],
    [-20, -50, 20, 20, -50, -20],
    [10, 20, 50, 50, 20, 10],
    [10, 20, 50, 50, 20, 10],
    [-20, -50, 20, 20, -50, -20],
    [100, -20, 10, 10, -20, 100],
]

def evaluate_board_with_table(board, stone, eval_table):
    score = 0
    for y in range(len(board)):
        for x in range(len(board[0])):
            if board[y][x] == stone:
                score += eval_table[y][x]
            elif board[y][x] == 3 - stone:
                score -= eval_table[y][x]
    return score

# X-squareペナルティを考慮した評価
def evaluate_with_risk(board, stone):
    x_squares = [(0, 1), (1, 0), (1, 1), (0, 4), (1, 5), (1, 4),
                 (4, 0), (5, 1), (4, 1), (4, 5), (5, 4), (4, 4)]
    score = evaluate_board_with_table(board, stone, EARLY_EVAL_TABLE)

    for x, y in x_squares:
        if board[y][x] == stone:
            score -= 100

    return score

# 置ける場所をリストアップする関数
def get_valid_moves(board, stone):
    moves = []
    for y in range(len(board)):
        for x in range(len(board[0])):
            if can_place_x_y(board, stone, x, y):
                moves.append((x, y))
    return moves

# 石を置く関数
def place_stone(board, stone, x, y):
    new_board = copy.deepcopy(board)
    new_board[y][x] = stone
    opponent = 3 - stone
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        flips = []
        while 0 <= nx < len(new_board[0]) and 0 <= ny < len(new_board) and new_board[ny][nx] == opponent:
            flips.append((nx, ny))
            nx += dx
            ny += dy
        if 0 <= nx < len(new_board[0]) and 0 <= ny < len(new_board) and new_board[ny][nx] == stone:
            for fx, fy in flips:
                new_board[fy][fx] = stone

    return new_board

# ミニマックス探索（改良版）
def minimax(board, stone, depth, is_maximizing):
    opponent = 3 - stone

    if depth == 0 or not can_place(board, stone) and not can_place(board, opponent):
        return evaluate_with_risk(board, stone)

    if is_maximizing:
        max_eval = -float('inf')
        for y in range(len(board)):
            for x in range(len(board[0])):
                if can_place_x_y(board, stone, x, y):
                    new_board = place_stone(board, stone, x, y)
                    eval = minimax(new_board, opponent, depth - 1, False)
                    max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for y in range(len(board)):
            for x in range(len(board[0])):
                if can_place_x_y(board, opponent, x, y):
                    new_board = place_stone(board, opponent, x, y)
                    eval = minimax(new_board, stone, depth - 1, True)
                    min_eval = min(min_eval, eval)
        return min_eval

# X-squareを避けつつ中盤の最善手を選択
def best_place_with_risk_management(board, stone):
    corners = [(0, 0), (0, 5), (5, 0), (5, 5)]
    x_squares = [(0, 1), (1, 0), (1, 1), (0, 4), (1, 5), (1, 4),
                 (4, 0), (5, 1), (4, 1), (4, 5), (5, 4), (4, 4)]

    best_score = -float('inf')
    best_move = None

    for y in range(len(board)):
        for x in range(len(board[0])):
            if not can_place_x_y(board, stone, x, y):
                continue

            if (x, y) in corners:
                return (x, y)

            score = count_flippable_stones(board, stone, x, y)

            if (x, y) in x_squares:
                score -= 100

            if score > best_score:
                best_score = score
                best_move = (x, y)

    return best_move

class ImprovedAI(object):

    def face(self):
        return "✨"

    def place(self, board, stone):
        empty_cells = sum(row.count(0) for row in board)
        if empty_cells <= 10:  # 終盤
            best_eval = -float('inf')
            best_move = None
            for y in range(len(board)):
                for x in range(len(board[0])):
                    if can_place_x_y(board, stone, x, y):
                        new_board = place_stone(board, stone, x, y)
                        eval = minimax(new_board, stone, depth=4, is_maximizing=False)
                        if eval > best_eval:
                            best_eval = eval
                            best_move = (x, y)
            return best_move

        else:  # 中盤
            best_score = -float('inf')
            best_move = None
            for y in range(len(board)):
                for x in range(len(board[0])):
                    if can_place_x_y(board, stone, x, y):
                        score = evaluate_with_risk(place_stone(board, stone, x, y), stone)
                        if score > best_score:
                            best_score = score
                            best_move = (x, y)
            return best_move
