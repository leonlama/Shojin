import chess
import pygame
from enum import Enum
import os
import random

# --- Constants ---
SCREEN_WIDTH = SCREEN_HEIGHT = 512
BOARD_SIZE = 8
SQUARE_SIZE = SCREEN_WIDTH // BOARD_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
SELECTED_SQUARE_COLOR = (255, 255, 0)
EVOLVED_HIGHLIGHT = (255, 215, 0)  # Gold color for evolved pieces
JINNED_HIGHLIGHT = (138, 43, 226)  # Purple for jinned pieces
MOVE_INDICATOR = (124, 252, 0)  # Light green for showing valid moves

class PieceState(Enum):
    NORMAL = 1
    EVOLVED = 2
    JINNED = 3

class PieceType(Enum):
    PAWN = 1    # Landsknecht
    BISHOP = 2  # Assassin
    KNIGHT = 3  # Shield Knight
    ROOK = 4    # Catapult
    QUEEN = 5   # Nemesis
    KING = 6    # Puppetmaster

# --- Image Loading ---
def load_piece_images():
    """Loads chess piece images from the assets folder."""
    piece_images = {}
    pieces = {
        'r': 'b_rook', 'n': 'b_knight', 'b': 'b_bishop',
        'q': 'b_queen', 'k': 'b_king', 'p': 'b_pawn',
        'R': 'w_rook', 'N': 'w_knight', 'B': 'w_bishop', 
        'Q': 'w_queen', 'K': 'w_king', 'P': 'w_pawn'
    }
    for piece, filename in pieces.items():
        img_path = os.path.join("assets", "imgs", f"{filename}.png")
        try:
            image = pygame.image.load(img_path)
            piece_images[piece] = pygame.transform.smoothscale(
                image, (SQUARE_SIZE, SQUARE_SIZE)
            )
        except pygame.error as e:
            print(f"Error loading image for {piece}: {e}")
    return piece_images

PIECE_IMAGES = load_piece_images()

# --- Sound Classes ---
class Sound:
    def __init__(self):
        self.move_sound = pygame.mixer.Sound("./assets/sounds/move_sound.mp3")
        self.evolve_sound = pygame.mixer.Sound("./assets/sounds/evolve_sound.mp3")
        self.ability_sound = pygame.mixer.Sound("./assets/sounds/ability_sound.mp3")

# --- Shojin Board ---
class ShojinBoard(chess.Board):
    def __init__(self, fen=chess.STARTING_FEN):  # Initialize with starting position
        super().__init__(fen)
        self.piece_states = {}  # Tracks piece states (NORMAL, EVOLVED, JINNED)
        self.jin_connections = []  # Tracks Jin connections (Puppetmaster -> Target)
        self._initialize_piece_states()

    def _initialize_piece_states(self):
        """Initialize back rank pieces as evolved, pawns as normal."""
        for square in chess.SQUARES:
            piece = self.piece_at(square)
            if piece:
                position = chess.square_name(square)
                piece_type = PieceType(piece.piece_type)
                # Check if piece is on back rank (rank 1 or 8)
                rank = chess.square_rank(square)
                is_back_rank = rank == 0 or rank == 7
                state = PieceState.EVOLVED if is_back_rank else PieceState.NORMAL
                self.piece_states[position] = (piece_type, state)

    def evolve_piece(self, position):
        """Evolve a piece if it's not already evolved."""
        if position in self.piece_states:
            piece_type, state = self.piece_states[position]
            if state == PieceState.NORMAL:
                self.piece_states[position] = (piece_type, PieceState.EVOLVED)
                return True
        return False

    def use_ability(self, position):
        """Use the ability of an evolved piece."""
        if position in self.piece_states:
            piece_type, state = self.piece_states[position]
            if state == PieceState.EVOLVED:
                if piece_type == PieceType.PAWN:  # Landsknecht
                    return self._landsknecht_ability(position)
                elif piece_type == PieceType.BISHOP:  # Assassin
                    return self._assassin_ability(position)
                elif piece_type == PieceType.ROOK:  # Catapult
                    return self._catapult_ability(position)
                elif piece_type == PieceType.QUEEN:  # Nemesis
                    return self._nemesis_ability(position)
                elif piece_type == PieceType.KING:  # Puppetmaster
                    return self._puppetmaster_ability(position)
        return False

    def _landsknecht_ability(self, position):
        """Landsknecht ability: two moves as a pawn."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if not piece or piece.piece_type != chess.PAWN:
            return False

        # Store original position
        start_square = square
        moves_made = 0
        valid_moves = []

        # Get valid pawn moves from current position
        for move in self.legal_moves:
            if move.from_square == square and not self.is_capture(move):
                valid_moves.append(move)

        # Make first move if valid moves exist
        if valid_moves:
            self.push(valid_moves[0])
            moves_made += 1
            
            # Get valid moves from new position
            square = valid_moves[0].to_square
            second_moves = []
            for move in self.legal_moves:
                if move.from_square == square and not self.is_capture(move):
                    second_moves.append(move)
                    
            # Make second move if possible
            if second_moves:
                self.push(second_moves[0])
                moves_made += 1

        # Degrade piece after ability use
        if moves_made > 0:
            self.piece_states[position] = (PieceType.PAWN, PieceState.NORMAL)
            return True
            
        return False

    def _assassin_ability(self, position):
        """Assassin ability: eliminate all enemy pieces in 8 surrounding squares."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if not piece or piece.piece_type != chess.BISHOP:
            return False

        # Get all 8 surrounding squares
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        surrounding_squares = []
        
        for f in range(max(0, file-1), min(7, file+2)):
            for r in range(max(0, rank-1), min(7, rank+2)):
                if (f,r) != (file,rank):
                    surrounding_squares.append(chess.square(f, r))

        # Remove enemy pieces in surrounding squares
        pieces_removed = False
        for s in surrounding_squares:
            target = self.piece_at(s)
            if target and target.color != piece.color:
                self.remove_piece_at(s)
                pieces_removed = True

        # Degrade piece after ability use
        if pieces_removed:
            self.piece_states[position] = (PieceType.BISHOP, PieceState.NORMAL)
            return True
            
        return False

    def _catapult_ability(self, position):
        """Catapult ability: eliminate first enemy in each rook-like direction."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if not piece or piece.piece_type != chess.ROOK:
            return False

        # Get all squares in rook directions
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        pieces_removed = False

        # Check each direction and remove first enemy found in each
        for d_file, d_rank in directions:
            curr_file, curr_rank = file + d_file, rank + d_rank
            while 0 <= curr_file <= 7 and 0 <= curr_rank <= 7:
                target_square = chess.square(curr_file, curr_rank)
                target = self.piece_at(target_square)
                if target:
                    if target.color != piece.color:
                        self.remove_piece_at(target_square)
                        pieces_removed = True
                    break
                curr_file += d_file
                curr_rank += d_rank

        # Degrade piece after ability use
        if pieces_removed:
            self.piece_states[position] = (PieceType.ROOK, PieceState.NORMAL)
            return True
            
        return False

    def _nemesis_ability(self, position):
        """Nemesis ability: degrade all enemy evolved pieces it could attack."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if not piece or piece.piece_type != chess.QUEEN:
            return False

        pieces_degraded = False
        # Get all squares the queen could attack
        for target_pos, (piece_type, state) in self.piece_states.items():
            target_square = chess.parse_square(target_pos)
            target = self.piece_at(target_square)
            
            if (target and target.color != piece.color and 
                state == PieceState.EVOLVED and
                self.is_attacked_by(piece.color, target_square)):
                # Degrade enemy evolved piece
                self.piece_states[target_pos] = (piece_type, PieceState.NORMAL)
                pieces_degraded = True

        # Degrade nemesis after ability use
        if pieces_degraded:
            self.piece_states[position] = (PieceType.QUEEN, PieceState.NORMAL)
            return True
            
        return False

    def _puppetmaster_ability(self, position):
        """Puppetmaster ability: Put king in check to create jin connection."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if not piece or piece.piece_type != chess.KING:
            return False

        # Check if king is putting himself in check
        for target_pos, (piece_type, state) in self.piece_states.items():
            target_square = chess.parse_square(target_pos)
            target = self.piece_at(target_square)
            
            if target and target.color != piece.color:
                # If king moves to a square attacked by this enemy piece
                if self.is_attacked_by(target.color, square):
                    # Create Jin connection
                    self.jin_connections.append((position, target_pos))
                    # Mark target piece as jinned
                    self.piece_states[target_pos] = (piece_type, PieceState.JINNED)
                    return True

        return False

# --- Improved Minimax AI ---
class MinimaxAI:
    def __init__(self, depth):
        self.depth = depth
        self.transposition_table = {}  # For caching positions
        self.piece_square_tables = self._initialize_piece_square_tables()
        self.move_ordering_scores = {}  # For move ordering

    def _initialize_piece_square_tables(self):
        """Initialize piece-square tables for positional evaluation."""
        tables = {}
        
        # Pawn/Landsknecht position scores - favor center control and advancement
        tables[chess.PAWN] = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]

        # Knight/Shield Knight position scores - favor center positions for protection
        tables[chess.KNIGHT] = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  5,  5,  5,  5,-20,-40,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30, 10, 20, 25, 25, 20, 10,-30,
            -30, 10, 20, 25, 25, 20, 10,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -40,-20,  5,  5,  5,  5,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50
        ]

        # Bishop/Assassin position scores - favor diagonals and center
        tables[chess.BISHOP] = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -10, 10, 10, 15, 15, 10, 10,-10,
            -10,  5, 15, 20, 20, 15,  5,-10,
            -10,  5, 15, 20, 20, 15,  5,-10,
            -10, 10, 10, 15, 15, 10, 10,-10,
            -10,  5,  5,  5,  5,  5,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20
        ]

        return tables

    def evaluate_board(self, board):
        """Enhanced board evaluation considering Shojin rules."""
        if board.is_checkmate():
            return -10000 if board.turn else 10000

        evaluation = 0
        
        # Material and state evaluation with Shojin considerations
        for pos, (piece_type, state) in board.piece_states.items():
            square = chess.parse_square(pos)
            piece = board.piece_at(square)
            if piece:
                value = self.get_piece_value(piece, state)
                
                # Add positional bonus
                if piece.piece_type in self.piece_square_tables:
                    table_value = self.piece_square_tables[piece.piece_type][square]
                    if not piece.color:  # If black, flip the table value
                        table_value = -table_value
                    value += table_value * 0.1

                # Special evaluation for Shield Knights (Schildritter)
                if piece.piece_type == chess.KNIGHT and state == PieceState.EVOLVED:
                    protected_pieces = self._count_protected_pieces(board, square, piece.color)
                    value += protected_pieces * 15  # Bonus for each protected piece

                # Evaluate Jin connections for Puppetmaster
                if piece.piece_type == chess.KING and state == PieceState.EVOLVED:
                    jin_value = self._evaluate_jin_connections(board, square)
                    value += jin_value

                evaluation += value if piece.color == chess.WHITE else -value

        # Mobility evaluation
        mobility = len(list(board.legal_moves))
        evaluation += 0.1 * mobility if board.turn else -0.1 * mobility

        # King safety with Shojin considerations
        w_king_square = board.king(chess.WHITE)
        b_king_square = board.king(chess.BLACK)
        if w_king_square:
            w_king_safety = self._evaluate_king_safety(board, w_king_square, chess.WHITE)
            evaluation += w_king_safety
        if b_king_square:
            b_king_safety = self._evaluate_king_safety(board, b_king_square, chess.BLACK)
            evaluation -= b_king_safety

        return evaluation

    def _evaluate_jin_connections(self, board, king_square):
        """Evaluate the value of Jin connections."""
        jin_value = 0
        for jin_connection in board.jin_connections:
            if jin_connection[0] == chess.square_name(king_square):
                target_square = chess.parse_square(jin_connection[1])
                target_piece = board.piece_at(target_square)
                if target_piece:
                    jin_value += self.get_piece_value(target_piece, PieceState.NORMAL) * 0.5
        return jin_value

    def _count_protected_pieces(self, board, shield_knight_square, color):
        """Count pieces protected by a Shield Knight."""
        protected = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                file = chess.square_file(shield_knight_square) + dx
                rank = chess.square_rank(shield_knight_square) + dy
                if 0 <= file <= 7 and 0 <= rank <= 7:
                    target_square = chess.square(file, rank)
                    piece = board.piece_at(target_square)
                    if piece and piece.color == color:
                        protected += 1
        return protected

    def get_piece_value(self, piece, state):
        """Piece valuation adapted for Shojin rules."""
        base_values = {
            chess.PAWN: 100,    # Landsknecht
            chess.KNIGHT: 320,  # Shield Knight
            chess.BISHOP: 330,  # Assassin
            chess.ROOK: 500,    # Catapult
            chess.QUEEN: 900,   # Nemesis
            chess.KING: 20000   # Puppetmaster
        }
        
        value = base_values.get(piece.piece_type, 0)
        
        # State multipliers adjusted for Shojin rules
        if state == PieceState.EVOLVED:
            if piece.piece_type == chess.KNIGHT:  # Shield Knight bonus
                value *= 2.0
            elif piece.piece_type == chess.QUEEN:  # Nemesis bonus
                value *= 1.8
            else:
                value *= 1.75
        elif state == PieceState.JINNED:
            value *= 0.5  # Jinned pieces are under opponent control
            
        return value

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax implementation adapted for Shojin rules."""
        # [Rest of the minimax implementation remains largely the same]
        # The evaluation changes are handled in evaluate_board()
        board_hash = str(board)
        if board_hash in self.transposition_table and self.transposition_table[board_hash][0] >= depth:
            return self.transposition_table[board_hash][1:]

        if depth == 0 or board.is_game_over():
            evaluation = self.evaluate_board(board)
            return evaluation, None

        moves = list(board.legal_moves)
        ordered_moves = self._order_moves(board, moves)
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()

                if evaluation > max_eval:
                    max_eval = evaluation
                    best_move = move
                    self.move_ordering_scores[move] = self.move_ordering_scores.get(move, 0) + depth

                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break

            self.transposition_table[board_hash] = (depth, max_eval, best_move)
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                evaluation, _ = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()

                if evaluation < min_eval:
                    min_eval = evaluation
                    best_move = move
                    self.move_ordering_scores[move] = self.move_ordering_scores.get(move, 0) + depth

                beta = min(beta, evaluation)
                if beta <= alpha:
                    break

            self.transposition_table[board_hash] = (depth, min_eval, best_move)
            return min_eval, best_move

    def find_best_move(self, board):
        """Find best move using iterative deepening."""
        best_move = None
        for current_depth in range(1, self.depth + 1):
            _, move = self.minimax(board, current_depth, float('-inf'), float('inf'), board.turn == chess.WHITE)
            if move:
                best_move = move
        return best_move
# --- Game ---
class Game:
    def __init__(self, depth=3):
        self.board = ShojinBoard()  # Will now initialize with starting position
        self.ai = MinimaxAI(depth)
        self.sound = Sound()
        self.selected_square = None
        self.game_over = False
        self.valid_moves = []  # Store valid moves for selected piece

    def make_ai_move(self):
        """Make a move using the AI."""
        if not self.game_over and not self.board.turn == chess.WHITE:
            best_move = self.ai.find_best_move(self.board)
            if best_move:
                self.board.push(best_move)
                self.sound.move_sound.play()

    def handle_mouse_click(self, pos):
        """Handle player moves."""
        row, col = self.get_square_from_mouse(pos)
        square = chess.square(col, 7 - row)
        
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.sound.move_sound.play()
                self.selected_square = None
                self.valid_moves = []  # Clear valid moves
            else:
                self.selected_square = None
                self.valid_moves = []  # Clear valid moves
        else:
            piece = self.board.piece_at(square)
            if piece and piece.color == chess.WHITE:
                self.selected_square = square
                # Get valid moves for selected piece
                self.valid_moves = [move for move in self.board.legal_moves 
                                  if move.from_square == square]

    def get_square_from_mouse(self, pos):
        """Convert mouse position to board coordinates."""
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def draw(self, screen):
        """Draw the board and pieces."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                # Draw move indicators
                square = chess.square(col, 7 - row)
                for move in self.valid_moves:
                    if move.to_square == square:
                        # Draw green dot
                        center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                        center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                        pygame.draw.circle(screen, MOVE_INDICATOR, (center_x, center_y), SQUARE_SIZE // 6)

                piece = self.board.piece_at(square)
                if piece:
                    # Draw evolved piece highlight
                    pos = chess.square_name(square)
                    if pos in self.board.piece_states:
                        _, state = self.board.piece_states[pos]
                        if state == PieceState.EVOLVED:
                            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                            s.set_alpha(128)
                            s.fill(EVOLVED_HIGHLIGHT)
                            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
                    
                    # Draw piece
                    image = PIECE_IMAGES[piece.symbol()]
                    screen.blit(image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

# --- Main Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Shojin Chess")

    game = Game(depth=3)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    game.handle_mouse_click(pygame.mouse.get_pos())

        game.make_ai_move()
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
