import chess
import pygame
from enum import Enum
import os

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
ABILITY_INDICATOR = (255, 0, 0)  # Red for showing ability targets
PROTECTED_HIGHLIGHT = (0, 191, 255)  # Deep sky blue for protected squares
JIN_HIGHLIGHT = (255, 0, 255)  # Magenta for jin connections

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

# --- Game ---
class Game:
    def __init__(self):
        self.board = chess.Board()
        self.sound = Sound()
        self.selected_square = None
        self.game_over = False
        self.valid_moves = []  # Store valid moves for selected piece
        self.ability_targets = []  # Store ability target squares
        self.piece_states = {}  # Track piece states
        self.protected_squares = set()  # Track squares protected by knight ability
        self.jinned_pieces = set()  # Track pieces under jin control
        self.jin_connections = []  # Track jin connections between puppetmaster and pieces
        self._initialize_piece_states()

    def _initialize_piece_states(self):
        """Initialize all pieces as evolved except pawns."""
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                position = chess.square_name(square)
                piece_type = PieceType(piece.piece_type)
                # Set pawns as normal, all other pieces as evolved
                if piece_type == PieceType.PAWN:
                    self.piece_states[position] = (piece_type, PieceState.NORMAL)
                else:
                    self.piece_states[position] = (piece_type, PieceState.EVOLVED)

    def get_square_from_mouse(self, pos):
        """Convert mouse coordinates to board row and column."""
        x, y = pos
        return (y // SQUARE_SIZE, x // SQUARE_SIZE)

    def evolve_piece(self, square):
        """Evolve a piece if conditions are met."""
        piece = self.board.piece_at(square)
        if not piece or piece.color != self.board.turn:
            return False
            
        pos = chess.square_name(square)
        if pos not in self.piece_states:
            return False
            
        piece_type, state = self.piece_states[pos]
        
        # Cannot evolve if in check
        if self.board.is_check():
            return False
            
        # Cannot evolve jinned pieces
        if pos in self.jinned_pieces:
            return False
            
        if state == PieceState.NORMAL:
            self.piece_states[pos] = (piece_type, PieceState.EVOLVED)
            self.sound.evolve_sound.play()
            
            # Initialize knight protection if evolved
            if piece_type == PieceType.KNIGHT:
                self._update_knight_protection(square)
                
            # End turn after evolving
            self.board.turn = not self.board.turn
                
            return True
            
        return False

    def handle_mouse_click(self, pos, right_click=False):
        """Handle player moves, abilities, and evolution."""
        row, col = self.get_square_from_mouse(pos)
        square = chess.square(col, 7 - row)
        
        if right_click:
            # Handle evolution
            if self.evolve_piece(square):
                return
                
            # Handle ability usage
            if self.selected_square is not None:
                if square in self.ability_targets:
                    self.use_ability(self.selected_square, square)
                    self.sound.ability_sound.play()
                self.selected_square = None
                self.ability_targets = []
        else:
            # Handle normal moves
            if self.selected_square is not None:
                move = chess.Move(self.selected_square, square)
                if move in self.board.legal_moves:
                    # Check if move is to a protected square
                    if square in self.protected_squares:
                        target_piece = self.board.piece_at(square)
                        if target_piece and target_piece.color != self.board.turn:
                            # Find and devolve the knight that protected this square
                            for knight_square in chess.SQUARES:
                                knight = self.board.piece_at(knight_square)
                                if knight and knight.piece_type == chess.KNIGHT:
                                    pos = chess.square_name(knight_square)
                                    if pos in self.piece_states:
                                        piece_type, state = self.piece_states[pos]
                                        if piece_type == PieceType.KNIGHT and state == PieceState.EVOLVED:
                                            self.piece_states[pos] = (piece_type, PieceState.NORMAL)
                                            self.protected_squares.clear()
                            self.selected_square = None
                            self.valid_moves = []
                            return
                            
                    # Make the move
                    self.board.push(move)
                    self.sound.move_sound.play()
                    
                    # Update piece states after move
                    old_pos = chess.square_name(self.selected_square)
                    new_pos = chess.square_name(square)
                    if old_pos in self.piece_states:
                        self.piece_states[new_pos] = self.piece_states.pop(old_pos)
                        
                        # Update knight protection if moved
                        piece_type, state = self.piece_states[new_pos]
                        if piece_type == PieceType.KNIGHT and state == PieceState.EVOLVED:
                            self.protected_squares.clear()
                            self._update_knight_protection(square)
                            
                    # Update jin connections
                    if old_pos in self.jinned_pieces:
                        self.jinned_pieces.remove(old_pos)
                        self.jinned_pieces.add(new_pos)
                        
                    # Check for pawn promotion
                    if self.board.piece_at(square).piece_type == chess.PAWN:
                        if chess.square_rank(square) in [0, 7]:
                            # Promote to queen by default
                            self.board.set_piece_at(square, chess.Piece(chess.QUEEN, self.board.turn))
                            self.piece_states[new_pos] = (PieceType.QUEEN, PieceState.NORMAL)
                            
                self.selected_square = None
                self.valid_moves = []
            else:
                piece = self.board.piece_at(square)
                if piece:
                    pos = chess.square_name(square)
                    # Can only move own pieces or jinned enemy pieces
                    if piece.color == self.board.turn or pos in self.jinned_pieces:
                        self.selected_square = square
                        self.valid_moves = [move for move in self.board.legal_moves 
                                          if move.from_square == square]
                        self.ability_targets = self.get_ability_targets(square)

    def get_ability_targets(self, square):
        """Get squares that can be targeted by piece's ability."""
        targets = []
        piece = self.board.piece_at(square)
        if not piece:
            return targets

        pos = chess.square_name(square)
        if pos not in self.piece_states:
            return targets

        piece_type, state = self.piece_states[pos]
        if state != PieceState.EVOLVED:
            return targets
            
        # Cannot use abilities if in check
        if self.board.is_check():
            return targets

        # Add ability target squares based on piece type
        if piece_type == PieceType.PAWN:  # Landsknecht
            # Add squares for double move
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            
            # Get possible moves in pawn direction
            direction = 1 if piece.color else -1
            for steps in range(1, 3):
                new_rank = rank + (steps * direction)
                if 0 <= new_rank <= 7:
                    target = chess.square(file, new_rank)
                    if not self.board.piece_at(target):
                        targets.append(target)
                    else:
                        break
                        
        elif piece_type == PieceType.BISHOP:  # Assassin
            # Add surrounding squares
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            for f in range(max(0, file-1), min(7, file+2)):
                for r in range(max(0, rank-1), min(7, rank+2)):
                    if (f,r) != (file,rank):
                        target = chess.square(f, r)
                        target_piece = self.board.piece_at(target)
                        if target_piece and target_piece.color != piece.color:
                            targets.append(target)
                            
        elif piece_type == PieceType.ROOK:  # Catapult
            # Add all squares that can be attacked in rook directions
            for direction in [(0,1), (0,-1), (1,0), (-1,0)]:
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                while True:
                    file += direction[0]
                    rank += direction[1]
                    if 0 <= file <= 7 and 0 <= rank <= 7:
                        target = chess.square(file, rank)
                        target_piece = self.board.piece_at(target)
                        if target_piece:
                            if target_piece.color != piece.color:
                                targets.append(target)
                            break
                    else:
                        break
                        
        elif piece_type == PieceType.QUEEN:  # Nemesis
            # Add all squares that can be attacked as queen
            for move in self.board.legal_moves:
                if move.from_square == square:
                    target = move.to_square
                    target_piece = self.board.piece_at(target)
                    if target_piece and target_piece.color != piece.color:
                        pos = chess.square_name(target)
                        if pos in self.piece_states:
                            piece_type, state = self.piece_states[pos]
                            if state == PieceState.EVOLVED:
                                targets.append(target)
                                
        elif piece_type == PieceType.KING:  # Puppetmaster
            # Add squares that would put king in check
            for move in self.board.legal_moves:
                if move.from_square == square:
                    target = move.to_square
                    target_piece = self.board.piece_at(target)
                    if target_piece and target_piece.color != piece.color:
                        # Test if move would create jin
                        self.board.push(move)
                        if self.board.is_check():
                            targets.append(target)
                        self.board.pop()

        return targets

    def use_ability(self, from_square, to_square):
        """Use piece ability."""
        piece = self.board.piece_at(from_square)
        if not piece:
            return False

        pos = chess.square_name(from_square)
        if pos not in self.piece_states:
            return False

        piece_type, state = self.piece_states[pos]
        if state != PieceState.EVOLVED:
            return False
            
        # Cannot use abilities if in check
        if self.board.is_check():
            return False

        # Handle ability effects
        if piece_type == PieceType.PAWN:  # Landsknecht
            # Make double move
            rank = chess.square_rank(from_square)
            file = chess.square_file(from_square)
            direction = 1 if piece.color else -1
            
            # First move
            new_rank = rank + direction
            if 0 <= new_rank <= 7:
                mid_square = chess.square(file, new_rank)
                if not self.board.piece_at(mid_square):
                    self.board.push(chess.Move(from_square, mid_square))
                    
                    # Second move
                    new_rank = new_rank + direction
                    if 0 <= new_rank <= 7:
                        final_square = chess.square(file, new_rank)
                        if not self.board.piece_at(final_square):
                            self.board.push(chess.Move(mid_square, final_square))
                            
        elif piece_type == PieceType.BISHOP:  # Assassin
            # Remove enemy pieces in surrounding squares
            file = chess.square_file(from_square)
            rank = chess.square_rank(from_square)
            for f in range(max(0, file-1), min(7, file+2)):
                for r in range(max(0, rank-1), min(7, rank+2)):
                    if (f,r) != (file,rank):
                        target = chess.square(f, r)
                        target_piece = self.board.piece_at(target)
                        if target_piece and target_piece.color != piece.color:
                            # Check if target is protected by Nemesis
                            target_pos = chess.square_name(target)
                            if target_pos in self.piece_states:
                                target_type, target_state = self.piece_states[target_pos]
                                if target_type == PieceType.QUEEN and target_state == PieceState.EVOLVED:
                                    # Nemesis is immune but gets devolved
                                    self.piece_states[target_pos] = (target_type, PieceState.NORMAL)
                                else:
                                    self.board.remove_piece_at(target)
                                    if target_pos in self.piece_states:
                                        del self.piece_states[target_pos]
                                        
        elif piece_type == PieceType.ROOK:  # Catapult
            # Remove enemy pieces in rook directions
            for direction in [(0,1), (0,-1), (1,0), (-1,0)]:
                file = chess.square_file(from_square)
                rank = chess.square_rank(from_square)
                while True:
                    file += direction[0]
                    rank += direction[1]
                    if 0 <= file <= 7 and 0 <= rank <= 7:
                        target = chess.square(file, rank)
                        target_piece = self.board.piece_at(target)
                        if target_piece:
                            if target_piece.color != piece.color:
                                # Check if target is protected by Nemesis
                                target_pos = chess.square_name(target)
                                if target_pos in self.piece_states:
                                    target_type, target_state = self.piece_states[target_pos]
                                    if target_type == PieceType.QUEEN and target_state == PieceState.EVOLVED:
                                        # Nemesis is immune but gets devolved
                                        self.piece_states[target_pos] = (target_type, PieceState.NORMAL)
                                    else:
                                        self.board.remove_piece_at(target)
                                        if target_pos in self.piece_states:
                                            del self.piece_states[target_pos]
                            break
                    else:
                        break
                        
        elif piece_type == PieceType.QUEEN:  # Nemesis
            # Devolve all evolved pieces that could be attacked as queen
            for move in self.board.legal_moves:
                if move.from_square == from_square:
                    target = move.to_square
                    target_piece = self.board.piece_at(target)
                    if target_piece and target_piece.color != piece.color:
                        target_pos = chess.square_name(target)
                        if target_pos in self.piece_states:
                            target_type, target_state = self.piece_states[target_pos]
                            if target_state == PieceState.EVOLVED:
                                self.piece_states[target_pos] = (target_type, PieceState.NORMAL)
                                
        elif piece_type == PieceType.KING:  # Puppetmaster
            # Create jin connection with target piece
            target_pos = chess.square_name(to_square)
            self.jinned_pieces.add(target_pos)
            self.jin_connections.append((from_square, to_square))

        # Degrade piece after ability use (except Puppetmaster and Shield Knight)
        if piece_type not in [PieceType.KING, PieceType.KNIGHT]:
            self.piece_states[pos] = (piece_type, PieceState.NORMAL)
            
        return True

    def draw(self, screen):
        """Draw the board and pieces."""
        # Draw board squares
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                # Draw move and ability indicators
                square = chess.square(col, 7 - row)
                if square in self.valid_moves:
                    center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    pygame.draw.circle(screen, MOVE_INDICATOR, (center_x, center_y), SQUARE_SIZE // 6)
                if square in self.ability_targets:
                    center_x = col * SQUARE_SIZE + SQUARE_SIZE // 2
                    center_y = row * SQUARE_SIZE + SQUARE_SIZE // 2
                    pygame.draw.circle(screen, ABILITY_INDICATOR, (center_x, center_y), SQUARE_SIZE // 6)

                # Draw piece state highlights
                piece = self.board.piece_at(square)
                if piece:
                    pos = chess.square_name(square)
                    if pos in self.piece_states:
                        _, state = self.piece_states[pos]
                        if state == PieceState.EVOLVED:
                            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                            s.set_alpha(128)
                            s.fill(EVOLVED_HIGHLIGHT)
                            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))
                        elif state == PieceState.JINNED:
                            s = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE))
                            s.set_alpha(128)
                            s.fill(JINNED_HIGHLIGHT)
                            screen.blit(s, (col * SQUARE_SIZE, row * SQUARE_SIZE))

                    # Draw the piece
                    image = PIECE_IMAGES[piece.symbol()]
                    screen.blit(image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

        # Draw jin connections
        for from_square, to_square in self.jin_connections:
            from_file = chess.square_file(from_square)
            from_rank = 7 - chess.square_rank(from_square)
            to_file = chess.square_file(to_square)
            to_rank = 7 - chess.square_rank(to_square)
            
            from_x = from_file * SQUARE_SIZE + SQUARE_SIZE // 2
            from_y = from_rank * SQUARE_SIZE + SQUARE_SIZE // 2
            to_x = to_file * SQUARE_SIZE + SQUARE_SIZE // 2
            to_y = to_rank * SQUARE_SIZE + SQUARE_SIZE // 2
            
            pygame.draw.line(screen, JIN_HIGHLIGHT, (from_x, from_y), (to_x, to_y), 2)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Shojin Chess")

    game = Game()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    game.handle_mouse_click(pygame.mouse.get_pos())
                elif event.button == 3:  # Right click
                    game.handle_mouse_click(pygame.mouse.get_pos(), right_click=True)

        game.draw(screen)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
