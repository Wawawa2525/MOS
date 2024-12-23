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

# 序盤専用のαβ探索
def alpha_beta_early(board, stone, depth, alpha, beta, maximizing):
    if depth == 0 or not can_place(board, stone):
        return evaluate_board_with_table(board, stone, EARLY_EVAL_TABLE), None

    moves = get_valid_moves(board, stone)
    moves = sorted(moves, key=lambda move: evaluate_board_with_table(place_stone(board, stone, move[0], move[1]), stone, EARLY_EVAL_TABLE), reverse=maximizing)

    if maximizing:
        max_eval = -float('inf')
        best_move = None
        for move in moves:
            new_board = place_stone(board, stone, move[0], move[1])
            eval_score, _ = alpha_beta_early(new_board, 3 - stone, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in moves:
            new_board = place_stone(board, stone, move[0], move[1])
            eval_score, _ = alpha_beta_early(new_board, 3 - stone, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move
def mcts_move(board, stone, simulations=2000):
    moves = get_valid_moves(board, stone)
    if not moves:
        return None

    def simulate_move(move):
        scores = 0
        for _ in range(simulations // len(moves)):
            simulated_board = place_stone(board, stone, move[0], move[1])
            current_stone = 3 - stone
            for _ in range(10):  # 最大10手のランダムプレイアウト
                valid_moves = get_valid_moves(simulated_board, current_stone)
                if not valid_moves:
                    break
                random_move = max(valid_moves, key=lambda m: evaluate_board_with_table(
                    place_stone(simulated_board, current_stone, m[0], m[1]),
                    current_stone,
                    MID_EVAL_TABLE
                ))
                simulated_board = place_stone(simulated_board, current_stone, random_move[0], random_move[1])
                current_stone = 3 - current_stone
            scores += evaluate_board_with_table(simulated_board, stone, MID_EVAL_TABLE)
        return scores

    with ThreadPoolExecutor() as executor:
        results = executor.map(simulate_move, moves)

    move_scores = {move: score for move, score in zip(moves, results)}
    best_move = max(move_scores, key=move_scores.get)
    return best_move

class HybridAI(object):

    def face(self):
        return "✨"

    def place(self, board, stone):
        valid_moves = get_valid_moves(board, stone)
        if not valid_moves:
            return None

        total_stones = sum(row.count(BLACK) + row.count(WHITE) for row in board)

        if total_stones < 20:  # 序盤
            _, best_move = alpha_beta_early(board, stone, depth=4, alpha=-float('inf'), beta=float('inf'), maximizing=True)
            return best_move
        elif total_stones < 50:  # 中盤
            return mcts_move(board, stone, simulations=2000)
        else:  # 終盤
            _, best_move = alpha_beta(board, stone, depth=8, alpha=-float('inf'), beta=float('inf'), maximizing=True)
            return best_move
