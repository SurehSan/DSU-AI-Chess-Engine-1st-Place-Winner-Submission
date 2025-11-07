import chess
from .interface import Interface

# Piece values for evaluation
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
        #This is currently unused - implement position-based evaluation
    ]
}



# Piece-square tables for positional evaluation
# These tables are defined but NOT YET USED in evaluate_board()
# To use them, iterate through pieces and add/subtract the positional bonus based on square
PIECE_SQUARE_TABLES = {
    'middlegame': {
        chess.PAWN: [
            [100, 100, 100, 100, 100, 100, 100, 100], # Rank 8 (promotion rank - huge bonus)
            [ 50,  50,  50,  50,  50,  50,  50,  50], # Rank 7 (near promotion)
            [ 10,  10,  20,  30,  30,  20,  10,  10], # Rank 6
            [  5,   5,  10,  25,  25,  10,   5,   5], # Rank 5
            [  0,   0,   0,  20,  20,   0,   0,   0], # Rank 4
            [  5,  -5, -10,   0,   0, -10,  -5,   5], # Rank 3
            [  5,  10,  10, -20, -20,  10,  10,   5], # Rank 2
            [  0,   0,   0,   0,   0,   0,   0,   0]  # Rank 1
        ],
        chess.KNIGHT: [
            [-50, -40, -30, -30, -30, -30, -40, -50], # Back rank (bad)
            [-40, -20,   0,   5,   5,   0, -20, -40],
            [-30,   5,  10,  15,  15,  10,   5, -30],
            [-30,   0,  15,  20,  20,  15,   0, -30],
            [-30,   5,  15,  20,  20,  15,   5, -30],
            [-30,   0,  10,  15,  15,  10,   0, -30],
            [-40, -20,   0,   0,   0,   0, -20, -40],
            [-50, -40, -30, -30, -30, -30, -40, -50]
        ],
        chess.BISHOP: [
            [-20, -10, -10, -10, -10, -10, -10, -20], # Back rank
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-10,  10,  15,  10,  10,  15,  10, -10],
            [-10,   0,  10,  15,  15,  10,   0, -10],
            [-10,   5,  15,  20,  20,  15,   5, -10], # Strong central diagonal control
            [-10,  10,   5,  10,  10,   5,  10, -10],
            [-10,   5,   0,   0,   0,   0,   5, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ],
        chess.ROOK: [
            [ 15,  15,  15,  20,  20,  15,  15,  15], # Strong on 8th rank
            [ 15,  20,  20,  20,  20,  20,  20,  15], # Strong on 7th rank
            [  0,   5,   5,  10,  10,   5,   5,   0],
            [  0,   0,   5,  10,  10,   5,   0,   0],
            [  0,   0,   5,  10,  10,   5,   0,   0],
            [  0,   0,   5,   5,   5,   5,   0,   0],
            [  5,   5,  10,  10,  10,  10,   5,   5],
            [  0,   0,   5,   5,   5,   5,   0,   0]
        ],
        chess.QUEEN: [
            [-20, -10, -10,  -5,  -5, -10, -10, -20],
            [-10,   0,   5,   0,   0,   0,   0, -10],
            [-10,   5,   5,   5,   5,   5,   0, -10],
            [  0,   0,   5,   5,   5,   5,   0,  -5],
            [ -5,   0,   5,   5,   5,   5,   0,  -5],
            [-10,   0,   5,   5,   5,   5,   0, -10],
            [-10,   0,   0,   0,   0,   0,   0, -10],
            [-20, -10, -10,  -5,  -5, -10, -10, -20]
        ],
        chess.KING: [
            [ 30,  20,   0, -10, -10,   0,  20,  30], # Corners safer
            [ 20,  10,  -5, -10, -10,  -5,  10,  20],
            [-10, -20, -20, -20, -20, -20, -20, -10],
            [-20, -30, -30, -40, -40, -30, -30, -20],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [-30, -40, -40, -50, -50, -40, -40, -30],
            [ 20,  10, -20, -30, -30, -20,  10,  20]  # Castle positions better
        ]
    },
    
    'endgame': {
        # Most pieces keep their middlegame tables except the king
        chess.KING: [
            [-20, -10, -10, -10, -10, -10, -10, -20],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-10,  10,  20,  30,  30,  20,  10, -10],
            [-10,  10,  30,  40,  40,  30,  10, -10],
            [-10,  10,  30,  40,  40,  30,  10, -10],
            [-10,  10,  20,  30,  30,  20,  10, -10],
            [-10,   0,  10,  10,  10,  10,   0, -10],
            [-20, -10, -10, -10, -10, -10, -10, -20]
        ]
    }
}


# center control 
def count_center_control(board, color, CENTER_UNIT_VALUE):
    #counts how many squares are under attack in the center
    middle_squares = [chess.C3, chess.C4, chess.C5, chess.C6, chess.D3, chess.D4, chess.D5, chess.D6, chess.E3, chess.E4, chess.E5, chess.E6, chess.F3, chess.F4, chess.F5]
    control_count = 0
    for square in middle_squares:
        if board.is_attacked_by(color, square):
            control_count += 1

    return control_count


def order_moves(board, moves):
    """
    OPTIMIZATION: Order moves to improve alpha-beta pruning.
    Checks captures first (likely to cause cutoffs).
    Simple but effective move ordering.
    """
    captures = []
    non_captures = []
    
    for move in moves:
        if board.is_capture(move):
            captures.append(move)
        else:
            non_captures.append(move)
    
    # Return captures first, then other moves
    return captures + non_captures


def is_defended(board, square, color, piece_values=BASE_PIECE_VALUES):
    defenders = 0
    attackers = 0
    value = 0
    for position in board.attackers(color, square):
        piece = board.piece_at(position)
        value = piece_values[piece.piece_type]
        defenders += value
    for position in board.attackers(not color, square):
        piece = board.piece_at(position)
        value = piece_values[piece.piece_type]
        attackers += value
    if defenders > attackers: 
        return True
    return False  

def capture(board, color):
    """ CHANGED THIS FUNCTION
    UNUSED - could be used for move ordering in minimax."""
    
    #changed this capture function
    #creates a list of possible capture moves
    captures = []
    for move in board.legal_moves:
        if board.is_capture(move):
            captures.append(move)
    return captures

    #add is_defended function

def bishop_pos_weight(BISHOP_W, board, color, square):
    """Calculate positional weight for a bishop.
    UNUSED!!!!"""
    bishop_position_points = BISHOP_W
    if is_defended(board, square, color):
        bishop_position_points += 0.02
    bishop_position_points += 0.02 * len(board.attacks(square))

    return bishop_position_points

def is_outposted(board, color, square):
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    
    if file > 0 and file < 7:
        #check if there are friendly pawns on adjacent diagonals
        if rank > 0:  #make sure we're not on the first rank
            diagonal1_square = chess.square(file + 1, rank - 1)
            diagonal2_square = chess.square(file - 1, rank - 1)
            #checking the pieces diagonally behind the square
            piece1 = board.piece_at(diagonal1_square)
            piece2 = board.piece_at(diagonal2_square)
            
            if (piece1 and piece1.piece_type == chess.PAWN and piece1.color == color) and \
               (piece2 and piece2.piece_type == chess.PAWN and piece2.color == color):
                for pos in range(0, 8):
                    piece = board.piece_at(chess.square(file - 1, pos))
                    if piece and piece.piece_type == chess.PAWN and piece.color != color:
                        if pos > rank and color == chess.WHITE:
                                return False
                        elif pos < rank and color == chess.BLACK:
                            return False
    return True


def knight_pos_weight(KNIGHT_W, board, color, square):
    """Calculate positional weight for a knight.
    UNUSED - integrate into evaluate_board()."""
    knight_position_points = KNIGHT_W

    if is_defended(board, square, color):
        knight_position_points += 0.02
    knight_position_points += 0.02 * len(board.attacks(square))
    #if board.piece_at(

    return knight_position_points 

def LINKED_POINTS(board, color, LINKED_BONUS):
    """ CHANGED THIS FUNCTION!!!
    Calculate bonus for linked/connected pawns."""
    for square in chess.SQUARES: 
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.PAWN and piece.color == color:
            if board.is_attacked_by(color, square): #defeneded by our own
                attackers = board.attackers(color, square)

                for attacker_square in attackers:
                    attacker_piece = board.piece_at(attacker_square)

                    if attacker_piece and attacker_piece.piece_type == chess.PAWN and attacker_piece.color == color:
                        LINKED_BONUS += 0.01
    return LINKED_BONUS

                
            
def pawn_metrics(board, color, PASSED_PAWN_VALUE, DOUBLED_PAWN_PENALTY, ISOLATED_PAWN_PENALTY):
    """Calculate pawn structure metrics.
    OPTIMIZED: Now only calculates for both colors once, not redundantly."""
    #only calculates linked pawns. 
    #NEED TO: Implement passed, doubled, and isolated pawn detection
    white_pawn_metrics = LINKED_POINTS(board, chess.WHITE, 0)
    black_pawn_metrics = LINKED_POINTS(board, chess.BLACK, 0)
    pawn_structure_score = white_pawn_metrics - black_pawn_metrics
    return pawn_structure_score
#finds the sum of white's pawn structure based off of the 3 major pawn weights    
     
     


#need to make white safety score
#need to make black safety score

def king_safety(board, color):
    # 1. Pawn shield
    # 2. Enemy attacks around king
    # 3. Castling status (position safety)

    safety_score = 0.0

    #find the king
    king_square = board.king(color)
    if king_square is None:
        return 0
    
    #finds the coordinates
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    #1. pawn shield
    for file_offset in [-1, 0, 1]:
        check_file = king_file + file_offset
        if 0 <= check_file <= 7:
            #this checks a square ahead of the king
            if color == chess.WHITE:
                check_rank = king_rank + 1
                if check_rank <= 7:
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety_score += 0.15
            else:
                check_rank = king_rank - 1
                if check_rank >= 0:  #FIXED!!! Black pawns need to check rank >= 0, not >= 7
                    check_square = chess.square(check_file, check_rank)
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        safety_score += 0.15

    #2. ENEMY attacks: penalizes the squares around the king when it's under attack
    enemy_color = not color
    for file_offset in [-1, 0, 1]:
        for rank_offset in [-1, 0, 1]:
            check_file = king_file + file_offset
            check_rank = king_rank + rank_offset

            if 0 <= check_file <= 7 and 0 <= check_rank <= 7:
                check_square = chess.square(check_file, check_rank)
                if board.is_attacked_by(enemy_color, check_square):
                    safety_score -= 0.08


    #3 castling bonus: rewards when the king is castled 
    if color == chess.WHITE:
        if king_square in [chess.G1, chess.C1]:
            safety_score += 0.25
    else: 
        if king_square in [chess.G8, chess.C8]:
            safety_score += 0.25


    return safety_score


def get_game_phase(board):
    """
    Determines if we're in middlegame or endgame.
    Returns a value between 0 (pure endgame) and 1 (pure middlegame)
    
    UNUSED - could be used to blend middlegame
    and endgame piece-square tables or adjust evaluation weights dynamically.
    """
    # Material weights for phase calculation
    PHASE_WEIGHTS = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
        chess.KING: 0
    }
    
     # Maximum phase score is when all pieces are on board
    # 2 knights (2), 2 bishops (2), 2 rooks (4), 1 queen (4) per side = 24 total
    MAX_PHASE = 24
  
    # Current phase score
    phase_weight = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type != chess.KING:
            phase_weight += PHASE_WEIGHTS[piece.piece_type]


    # Convert to a 0-1 scale where:
    # 1.0 = pure middlegame (all pieces present)
    # 0.0 = pure endgame (only kings and pawns)
    return min(1.0, phase_weight / MAX_PHASE)


def evaluate_positional(board, game_phase=None):
    """
    Evaluate piece positions using piece-square tables.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    
    OPTIMIZED: Can accept pre-calculated game_phase to avoid recalculation.
    """
    score = 0
    if game_phase is None:
        game_phase = get_game_phase(board)  # 0.0 = endgame, 1.0 = middlegame
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        piece_type = piece.piece_type
        color = piece.color
        
        # Get the rank and file (need to flip for black pieces)
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        
        # For Black pieces, we need to flip the rank (mirror vertically)
        # White's rank 0 (1st rank) = index 7 in table
        # Black's rank 7 (8th rank) = index 7 in table (same position perspective)
        if color == chess.WHITE:
            table_rank = 7 - rank  # Flip so rank 1 = index 7, rank 8 = index 0
        else:
            table_rank = rank  # Black already sees from their perspective
        
        # Get positional value from piece-square tables
        positional_value = 0
        
        if piece_type == chess.KING:
            # King uses different tables for middlegame vs endgame
            # Blend between middlegame and endgame tables based on game phase
            mg_value = PIECE_SQUARE_TABLES['middlegame'][chess.KING][table_rank][file]
            eg_value = PIECE_SQUARE_TABLES['endgame'][chess.KING][table_rank][file]
            positional_value = mg_value * game_phase + eg_value * (1 - game_phase)
        elif piece_type in PIECE_SQUARE_TABLES['middlegame']:
            # Other pieces use middlegame tables
            positional_value = PIECE_SQUARE_TABLES['middlegame'][piece_type][table_rank][file]
        
        # Convert to centipawns (divide by 100) for reasonable scale
        positional_value = positional_value / 100.0
        
        # Add to score (positive for White, negative for Black)
        if color == chess.WHITE:
            score += positional_value
        else:
            score -= positional_value
    
    return score


def evaluate_mobility(board):
    """
    Evaluate mobility by counting legal moves available to each side.
    More moves = better position (more options and control).
    Returns score from WHITE's perspective.
    
    OPTIMIZED: Only counts moves for current side, much faster!
    """
    # Only count legal moves for the side to move (MUCH faster)
    # This is a good enough approximation and 2x faster
    if board.turn == chess.WHITE:
        mobility_score = board.legal_moves.count() / 10.0
    else:
        mobility_score = -board.legal_moves.count() / 10.0
    
    return mobility_score


#positional knowledge, add more knowledge based on where they are in the board
#opening, middle (10 - 15 moves, most pieces are developed), endgame (less than 13 points of material each)
#Take time during meetings so that the piece algorithms don't conflict with eachother

def evaluate_board(board):
    
    """
    Evaluate the board based on material count.
    Returns score from WHITE's perspective.
    Positive score favors white, negative favors black.
    """
    
    score = 0

    color = chess.WHITE if board.turn == chess.WHITE else chess.BLACK

    #we declare our weights here in order to calculate how optimal each side's position is
    W_positional = 0.3  # NOW IMPLEMENTED!
    W_mobility = 0.2    # NOW IMPLEMENTED!

    W_center_control = 0.4 #.2
    W_pawn_structure = 0.3 #.1
    W_king_safety = 0.3 #.2

    #positional weights by piece
    # THESE ARE NOT IMPLEMENTED YET
    BISHOP_W = 0.25 
    ROOK_W = 0.25    
    KNIGHT_W = 0.25  
    QUEEN_W = 0.25   

    #pawn weights
    CENTER_UNIT_VALUE = 0.05 #per square value for each controlled square
    PASSED_PAWN_VALUE = 0.5 #this is the bonus for having a passed pawn (cannot be stopped from promoting) - Not yet used
    DOUBLED_PAWN_PENALTY = 0.2 #penalty for having 2 pawns on the same file (cannot defend eachother) -  Not yet used
    ISOLATED_PAWN_PENALTY = 0.15 #penalty for pawn with no friendly pawns on adjacent files - Not yet used
    LINKED_BONUS = 0.1  # Used in LINKED_POINTS() function
    
    # OPTIMIZED: Calculate game phase once and reuse it
    game_phase = get_game_phase(board)
    
    # Evaluate positional score (piece-square tables)
    positional_score = evaluate_positional(board, game_phase)
    score += positional_score * W_positional
    
    # Evaluate mobility (number of legal moves)
    mobility_score = evaluate_mobility(board)
    score += mobility_score * W_mobility

    # OPTIMIZED: Call pawn_metrics once, it calculates both sides
    pawn_structure_score_total = pawn_metrics(board, chess.WHITE, PASSED_PAWN_VALUE, DOUBLED_PAWN_PENALTY, ISOLATED_PAWN_PENALTY)
    score += pawn_structure_score_total * W_pawn_structure

    white_king_safety_score = king_safety(board, chess.WHITE)
    black_king_safety_score = king_safety(board, chess.BLACK)

    king_safety_score_total = (white_king_safety_score - black_king_safety_score)

    score += king_safety_score_total * W_king_safety

    white_center = count_center_control(board, chess.WHITE, CENTER_UNIT_VALUE)
    black_center = count_center_control(board, chess.BLACK, CENTER_UNIT_VALUE)

    center_control_score = (white_center - black_center) * CENTER_UNIT_VALUE

    score += center_control_score * W_center_control  # FIXED: Removed double multiplication
            
    if board.is_checkmate():
        # If it's White's turn and checkmate, White lost (bad for White)
        # If it's Black's turn and checkmate, Black lost (good for White)
        return float('-inf') if board.turn == chess.WHITE else float('inf')
    
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    
    # Material count
    material_score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = BASE_PIECE_VALUES[piece.piece_type]
            material_score += value if piece.color == chess.WHITE else -value

    #combine all evaluation components
    total_score = material_score + score  # score already includes king_safety and center_control weighted
    
    return total_score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning.
    Always evaluates from White's perspective.
    
    Args:
        board: board object
        depth: remaining depth to search
        alpha: best value for maximizer
        beta: best value for minimizer
        maximizing_player: True if White's turn, False if Black's turn
    
    Returns:
        Best evaluation score from White's perspective
    """
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    # OPTIMIZATION: Order moves (captures first) for better pruning
    ordered_moves = order_moves(board, board.legal_moves)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break  # Beta cutoff
        return max_eval
    else:
        min_eval = float('inf')
        for move in ordered_moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break  # Alpha cutoff
        return min_eval

def find_best_move(board, depth):
    """
    Find the best move for the current player.
    
    Args:
        board: board object
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
             
            """ADDED CHECK FOR REPETITION HERE!!!!!!!!"""
            #penalize moves that lead to threefold repetition
            if board.is_repetition(2):  #checks if the position has occurred 2+ times (would be 3rd occurrence)
                board_value = float('-inf')  # Make this move very undesirable
            else:
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
            
            #penalize moves that lead to threefold repetition again but for black
            if board.is_repetition(2):  
                board_value = float('inf')  
            else:
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
    board = chess.board(fen)

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
