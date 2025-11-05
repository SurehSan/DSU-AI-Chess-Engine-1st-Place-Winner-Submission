import chess
from .interface import Interface


# Piece values for evaluation
# Create multiple sections for piece values based on position and proximity to other pieces
# and how that value would change based on what type of piece that piece is near
BASE_PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,

    chess.BISHOP: 3,
    chess.ROOK: 5,

    chess.QUEEN: 9,
    chess.KING: 0
}

POSITION_TABLES = {
    chess.PAWN: [
        # 8x8 grid with position values
        # Center pawns are worth more
        # Advancing pawns worth progressively more
    ]
}



PIECE_SQUARE_TABLES = {
    'middlegame': {
        chess.PAWN: [
            # Centered pawns worth more
            # 8x8 grid with gradual increase in values as pawns reach the center
        ],
        chess.KNIGHT: [
            # Centered knights worth more, worth less on edges
            # 8x8 grid emphasizing center control
        ],
        chess.BISHOP: [
            # Strong on long diagonals
            # 8x8 grid emphasizing long diagonals
        ],
        chess.ROOK: [
            # Strong on 7th rank and open files
            # 8x8 grid emphasizing file control
        ],
        chess.QUEEN: [
            # Queen slightly better in center, should have good mobility in general
            # 8x8 grid emphasizing castling
        ],
        chess.KING: [
            # King safety in early/middle game
            # 8x8 grid emphas
        ]
    },
    'endgame': {
        # Similar structure with different values
        # Especially for the king as they become more active
        # Pawns also become more valuable when available to promotion
    }
}

#Here, we will add weights based on how much control a piece has, what type of safety each piece is in, etc

W_positional = 0.3
W_mobility = 0.2
W_center_control = 0.2
W_pawn_structure = 0.1
W_king_safety = 0.2

CENTER_UNIT_VALUE = 0.05 #per square value for each controlled square
PASSED_PAWN_VALUE = 0.5 #this is the bonus for having a passed pawn (cannot be stopped from promoting)
DOUBLED_PAWN_PENALTY = 0.2 #penalty for having 2 pawns on the same file (cannot defend eachother )
ISOLATED_PAWN_PENALTY = 0.15 #penalty for pawn with no friendly pawns on adjacent files

Total_W = W_positional + W_mobility + W_center_control + W_pawn_structure + W_king_safety + CENTER_UNIT_VALUE + PASSED_PAWN_VALUE - DOUBLED_PAWN_PENALTY - ISOLATED_PAWN_PENALTY




# center control 
def count_center_control(board, color):
    #counts how many squares are under attack in the center
    middle_squares = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D4, chess.D5, chess.D6, chess.E3, chess.E4, chess.E5, chess.E6, chess.F3, chess.F4, chess.F5]
    control_count = 0
    for square in middle_squares:
        if board.is_attacked_by(color, square):
            control_count += 1

    return control_count

white_center = count_center_control(chess.board, chess.WHITE)
black_center = count_center_control(chess.board, chess.BLACK)
center_control_score = (white_center - black_center) * CENTER_UNIT_VALUE

def capture(board, color):
    for square in chess.SQUARES():
        if board.attackers(square, )

def pawn_structure_metrics(board, color):
    pawn_structure_points = W_pawn_structure
    if color == chess.WHITE:
        for square in chess.SQUARES:
            if board.piece_at_square() == chess.PAWN:
                if board.is_attacked_by(chess.WHITE, square):
                    attackers = board.attackers(chess.WHITE, square)
                    for piece in attackers:
                        if piece == chess.PAWN and piece.color == chess.WHITE:
                            pawn_structure_points += 0.01
    if color == chess.BLACK:
        for square in chess.SQUARES:
            if board.piece_at_square() == chess.PAWN:
                if board.is_attacked_by(chess.BLACK, square):
                    attackers = board.attackers(chess.BLACK, square)
                    for piece in attackers:
                        if piece == chess.PAWN and piece.color == chess.BLACK:
                            pawn_structure_points -= 0.01
    return pawn_structure_points
                
            
white_pawn_metrics = pawn_structure_metrics(chess.board, chess.WHITE)
black_pawn_metrics = pawn_structure_metrics(chess.board, chess.BLACK)

#sums all of the pawn features with penalties and values
pawn_structure_score = (white_pawn_metrics.passed * PASSED_PAWN_VALUE - white_pawn_metrics.doubled * DOUBLED_PAWN_PENALTY - white_pawn_metrics.isolated * ISOLATED_PAWN_PENALTY)
pawn_structure_score -= (black_pawn_metrics.passed * PASSED_PAWN_VALUE - black_pawn_metrics.doubled * DOUBLED_PAWN_PENALTY - black_pawn_metrics.isolated * ISOLATED_PAWN_PENALTY)
#finds the sum of white's pawn structure based off of the 3 major pawn weights          

white_king_safety = king_safety_score(chess.board, chess.WHITE)
black_king_safety = black_safety_score(chess.board, chess.BLACK)
king_safety_score_total = white_king_safety - black_king_safety




#positional knowledge, add more knowledge based on where they are in the board
#opening, middle (10 - 15 moves, most pieces are developed), endgame (less than 13 points of material each)
#Take time during meetings so that the piece algorithms don't conflict with eachother

def evaluate_board(board):
    
    """
    Evaluate the board based on material count.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    if board.can_claim_threefold_repetition():
        if board.turn == chess.WHITE:
            score = float("-inf")
            return score
        elif board.turn == chess.BLACK:
            score = float("inf")
            return score
            
    if board.is_checkmate():
        # If it's White's turn and checkmate, White lost (bad for White)
        # If it's Black's turn and checkmate, Black lost (good for White)
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = BASE_PIECE_VALUES[piece.piece_type]
            score += value if piece.color == chess.WHITE else -value
            
    score = score * Total_W

        
    
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning.
    Always evaluates from White's perspective.
    
    Args:
        board: chess.Board object
        depth: remaining depth to search
        alpha: best value for maximizer
        beta: best value for minimizer
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        Best evaluation score from White's perspective
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def find_best_move(board, depth):
    """
    Find the best move for the current player.
    
    Args:
        board: chess.Board object
        depth: search depth
    
    Returns:
        Best move in UCI notation (e.g., 'e2e4')
    """
    best_move = None
    alpha = float('-inf')
    beta = float('inf')
    
    if board.turn == chess.WHITE:
        # White wants to MAXIMIZE the score
        best_value = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            
            if board_value > best_value:
                best_value = board_value
                best_move = move
            alpha = max(alpha, best_value)
    else:
        # Black wants to MINIMIZE the score
        best_value = float('inf')
        for move in board.legal_moves:
            board.push(move)
            board_value = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            
            if board_value < best_value:
                best_value = board_value
                best_move = move
            beta = min(beta, best_value)
    
    return best_move

def play(interface: Interface, color = "w"):
    search_depth = 4  # Can be any positive number
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    board = chess.Board(fen)

    # color = interface.input()

    if color == "b":
        move = interface.input()
        board.push_san(move)

    while True:
        best_move = find_best_move(board, search_depth)
        interface.output(board.san(best_move))
        board.push(best_move)

        move = interface.input()
        board.push_san(move)
        # print(board)
