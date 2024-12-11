import chess
import chess.engine
import pygame
from enum import Enum

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
EVOLVED_HIGHLIGHT = (255, 215, 0) # Gold color for evolved pieces
JINNED_HIGHLIGHT = (138, 43, 226) # Purple for jinned pieces

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
    pieces = ['r', 'n', 'b', 'q', 'k', 'p',
              'R', 'N', 'B', 'Q', 'K', 'P']
    imgs = {
        'r': 'b_rook',
        'n': 'b_knight', 
        'b': 'b_bishop',
        'q': 'b_queen',
        'k': 'b_king',
        'p': 'b_pawn',
        'R': 'w_rook',
        'N': 'w_knight',
        'B': 'w_bishop',
        'Q': 'w_queen',
        'K': 'w_king',
        'P': 'w_pawn',
    }
    evolved_imgs = {
        'r': 'b_catapult',
        'n': 'b_shield_knight',
        'b': 'b_assassin', 
        'q': 'b_nemesis',
        'k': 'b_puppetmaster',
        'p': 'b_landsknecht',
        'R': 'w_catapult',
        'N': 'w_shield_knight',
        'B': 'w_assassin',
        'Q': 'w_nemesis',
        'K': 'w_puppetmaster',
        'P': 'w_landsknecht',
    }
    
    for piece in pieces:
        # Load normal piece image
        img_path = f"assets/imgs/{imgs[piece]}.png"
        piece_images[piece] = pygame.image.load(img_path)
        piece_images[piece] = pygame.transform.smoothscale(
            piece_images[piece], (SQUARE_SIZE, SQUARE_SIZE))
            
        # Load evolved piece image
        evolved_path = f"assets/imgs/{evolved_imgs[piece]}.png"
        piece_images[f"{piece}_evolved"] = pygame.image.load(evolved_path)
        piece_images[f"{piece}_evolved"] = pygame.transform.smoothscale(
            piece_images[f"{piece}_evolved"], (SQUARE_SIZE, SQUARE_SIZE))
            
    return piece_images

PIECE_IMAGES = load_piece_images()

# --- Sound Classes ---
class Sound:
    def __init__(self):
        self.check_sound = pygame.mixer.Sound("./assets/sounds/check_sound.mp3")
        self.game_over_sound = pygame.mixer.Sound("./assets/sounds/gameover_sound.mp3")
        self.game_start_sound = pygame.mixer.Sound("./assets/sounds/start_sound.mp3")
        self.move_sound = pygame.mixer.Sound("./assets/sounds/move_sound.mp3")
        self.stalemate_sound = pygame.mixer.Sound("./assets/sounds/stalemate_sound.mp3")
        self.evolve_sound = pygame.mixer.Sound("./assets/sounds/evolve_sound.mp3")
        self.ability_sound = pygame.mixer.Sound("./assets/sounds/ability_sound.mp3")
        self.jin_sound = pygame.mixer.Sound("./assets/sounds/jin_sound.mp3")

# --- Game Classes ---
class ShojinBoard(chess.Board):
    def __init__(self):
        super().__init__()
        self.piece_states = {} # Maps positions to (PieceType, PieceState)
        self.jin_connections = [] # List of (puppetmaster_pos, jinned_piece_pos) tuples
        self.jinned_piece_owners = {} # Maps jinned piece positions to original owner
        self._initialize_piece_states()
        
    def _initialize_piece_states(self):
        """Initialize piece states for all pieces on the board."""
        for square in chess.SQUARES:
            piece = self.piece_at(square)
        if piece is not None:
            pos = chess.square_name(square)
            piece_type = PieceType(piece.piece_type)
            if piece_type == PieceType.PAWN:
                self.piece_states[pos] = (piece_type, PieceState.NORMAL)
            else:
                self.piece_states[pos] = (piece_type, PieceState.EVOLVED)

    def evolve_piece(self, position: str) -> bool:
        """Evolve a piece if possible. Returns True if successful."""
        if position not in self.piece_states:
            return False
            
        piece_type, state = self.piece_states[position]
        if state != PieceState.NORMAL:
            return False

        # Update piece state to evolved
        self.piece_states[position] = (piece_type, PieceState.EVOLVED)
        return True

    def use_evolution_ability(self, position: str) -> bool:
        """Use the evolution ability of a piece if available.
        Returns True if successful, False otherwise."""
        if position not in self.piece_states:
            return False
            
        piece_type, state = self.piece_states[position]
        if state != PieceState.EVOLVED:
            return False

        # Handle different evolution abilities
        if piece_type == PieceType.PAWN:
            # Landsknecht double move logic
            file, rank = position[0], int(position[1])
            forward = 1 if self.turn else -1
            target_rank = rank + (2 * forward)
            if 1 <= target_rank <= 8:
                target_pos = f"{file}{target_rank}"
                if target_pos not in self.piece_states:
                    self._make_move(f"{position}{target_pos}")
                    return True
            return False
            
        elif piece_type == PieceType.BISHOP:
            # Assassin area attack logic - captures all pieces in 3x3 area
            file, rank = ord(position[0]) - ord('a'), int(position[1]) - 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    new_file = chr(file + ord('a') + dx)
                    new_rank = rank + dy + 1
                    if 'a' <= new_file <= 'h' and 1 <= new_rank <= 8:
                        target = f"{new_file}{new_rank}"
                        if target in self.piece_states:
                            self._capture_piece(target)
            return True
            
        elif piece_type == PieceType.ROOK:
            # Catapult ranged attack logic - captures first piece in each rook direction
            file, rank = ord(position[0]) - ord('a'), int(position[1]) - 1
            directions = [(0,1), (0,-1), (1,0), (-1,0)]  # up, down, right, left
            for dx, dy in directions:
                # Check each square in the direction until board edge
                steps = 1
                while True:
                    new_file = chr(file + ord('a') + (dx * steps))
                    new_rank = rank + (dy * steps) + 1
                    if not ('a' <= new_file <= 'h' and 1 <= new_rank <= 8):
                        break
                    target = f"{new_file}{new_rank}"
                    if target in self.piece_states:
                        self._capture_piece(target)
                        break
                    steps += 1
            return True
            
        elif piece_type == PieceType.QUEEN:
            # Nemesis ability - can move to any square occupied by an enemy piece
            for pos, (p_type, p_state) in self.piece_states.items():
                if pos != position and self._is_enemy_piece(pos):
                    self._make_move(f"{position}{pos}")
                    return True
            return False
            
        elif piece_type == PieceType.KING:
            # Puppetmaster jin logic - handled by apply_jin method
            return self._initiate_jin_mode()

        # Revert to normal state after using ability
        self.piece_states[position] = (piece_type, PieceState.NORMAL)
        return True

    def apply_jin(self, puppetmaster_pos: str, target_pos: str) -> bool:
        """Apply jin control from puppetmaster to target piece."""
        if not self._is_valid_jin(puppetmaster_pos, target_pos):
            return False
            
        # Remove any existing jin connections for this puppetmaster
        self.jin_connections = [(pm, target) for pm, target in self.jin_connections 
                               if pm != puppetmaster_pos]
            
        # Add the new jin connection
        self.jin_connections.append((puppetmaster_pos, target_pos))
        
        # Update piece state to jinned
        piece_type = self.piece_states[target_pos][0]
        self.piece_states[target_pos] = (piece_type, PieceState.JINNED)
        
        # Track original owner for restoration if jin breaks
        self.jinned_piece_owners[target_pos] = not self.turn
        
        return True

    def _is_valid_jin(self, puppetmaster_pos: str, target_pos: str) -> bool:
        """Check if jin application is valid."""
        if puppetmaster_pos not in self.piece_states or target_pos not in self.piece_states:
            return False
            
        pm_type, pm_state = self.piece_states[puppetmaster_pos]
        if pm_type != PieceType.KING or pm_state != PieceState.EVOLVED:
            return False
            
        # Target must be enemy piece and not already jinned
        target_type, target_state = self.piece_states[target_pos]
        if not self._is_enemy_piece(target_pos) or target_state == PieceState.JINNED:
            return False
            
        return True

    def _is_enemy_piece(self, position: str) -> bool:
        """Check if piece at position belongs to enemy."""
        square = chess.parse_square(position)
        piece = self.piece_at(square)
        if piece is None:
            return False
        return piece.color != self.turn

    def _make_move(self, move_str: str):
        """Make a move on the board."""
        move = chess.Move.from_uci(move_str)
        # Update piece states when making a move
        from_square = move_str[:2]
        to_square = move_str[2:]
        if from_square in self.piece_states:
            self.piece_states[to_square] = self.piece_states[from_square]
            del self.piece_states[from_square]
        self.push(move)

    def _capture_piece(self, position: str):
        """Remove a piece from the board."""
        square = chess.parse_square(position)
        self.remove_piece_at(square)
        if position in self.piece_states:
            del self.piece_states[position]

    def _initiate_jin_mode(self) -> bool:
        """Initiate jin mode for puppetmaster."""
        # Implementation depends on UI interaction
        return True

# --- Game Classes ---
class Game:
    def __init__(self, engine_path, player_color=chess.WHITE):
        self.engine_path = engine_path
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.board = ShojinBoard()
        self.player_color = player_color
        self.selected_square = None
        self.game_over = False
        self.sound = Sound()
        self.sound.game_start_sound.play()
        self.evolution_mode = False
        self.jin_mode = False

    def reset(self):
        """Resets the game state."""
        self.sound.game_start_sound.play()
        self.board = ShojinBoard()
        self.selected_square = None
        self.game_over = False
        self.evolution_mode = False
        self.jin_mode = False

    def handle_mouse_click(self, pos):
        """Handles mouse clicks to select and move pieces."""
        if not self.game_over and self.board.turn == self.player_color:
            row, col = self.get_square_from_mouse(pos)
            square = chess.square(col, 7 - row)
            pos_str = chess.square_name(square)

            if self.evolution_mode:
                if self.board.evolve_piece(pos_str):
                    self.sound.evolve_sound.play()
                self.evolution_mode = False
                self.selected_square = None
                return

            if self.jin_mode:
                if self.selected_square:
                    from_square = chess.square_name(chess.square(
                        self.selected_square[1], 7 - self.selected_square[0]))
                    if self.board.apply_jin(from_square, pos_str):
                        self.sound.jin_sound.play()
                    self.jin_mode = False
                    self.selected_square = None
                return

            if self.selected_square is None:
                # Select a piece if it belongs to the current player
                if self.board.piece_at(square) is not None and self.board.color_at(square) == self.player_color:
                    self.selected_square = (row, col)
            else:
                # Try to make a move
                from_square = chess.square_name(chess.square(
                    self.selected_square[1], 7 - self.selected_square[0]))
                move = chess.Move(chess.square(
                    self.selected_square[1], 7 - self.selected_square[0]), square)
                
                # Check for evolution ability use
                if from_square == pos_str and self.board.piece_states[from_square][1] == PieceState.EVOLVED:
                    if self.board.use_evolution_ability(from_square):
                        self.sound.ability_sound.play()
                elif move in self.board.legal_moves:
                    self.board.push(move)
                    self.sound.move_sound.play()
                
                self.selected_square = None

    def get_square_from_mouse(self, pos):
        """Converts mouse coordinates to chessboard square coordinates."""
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def make_ai_move(self):
        """Makes a move using the Stockfish engine."""
        if not self.game_over and self.board.turn != self.player_color:
            result = self.engine.play(self.board, chess.engine.Limit(time=1))
            self.board.push(result.move)
            self.sound.move_sound.play()

    def update_game_state(self):
        """Checks for game over conditions."""
        if self.board.is_checkmate() or self.board.is_stalemate():
            self.game_over = True

    def draw(self, screen):
        """Draws the board, pieces, and game over messages."""
        self.draw_board(screen)
        if self.selected_square:
            self.highlight_moves(screen)

        if self.game_over:
            self.sound.game_over_sound.play()
            self.draw_game_over(screen)

    def draw_board(self, screen):
        """Draws the chessboard."""
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
                pygame.draw.rect(screen, color, (col * SQUARE_SIZE,
                                                 row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

                # Highlight selected square
                if self.selected_square is not None and (row, col) == self.selected_square:
                    pygame.draw.rect(screen, SELECTED_SQUARE_COLOR, (col *
                                     SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

                # Draw pieces
                square = chess.square(col, 7 - row)
                pos_str = chess.square_name(square)
                piece = str(self.board.piece_at(square))
                
                if piece != 'None':
                    # Get piece state if it exists, otherwise use normal state
                    piece_state = PieceState.NORMAL
                    if pos_str in self.board.piece_states:
                        piece_state = self.board.piece_states[pos_str][1]
                    
                    # Choose image based on state
                    if piece_state == PieceState.EVOLVED:
                        image = PIECE_IMAGES[f"{piece}_evolved"]
                        pygame.draw.rect(screen, EVOLVED_HIGHLIGHT, (col *
                                         SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
                    elif piece_state == PieceState.JINNED:
                        image = PIECE_IMAGES[piece]
                        pygame.draw.rect(screen, JINNED_HIGHLIGHT, (col *
                                         SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 2)
                    else:
                        image = PIECE_IMAGES[piece]
                        
                    screen.blit(image, (col * SQUARE_SIZE, row * SQUARE_SIZE))

    def highlight_moves(self, screen):
        """Highlights legal moves for the selected piece."""
        if self.selected_square:
            possible_moves = self.get_possible_moves()
            for move in possible_moves:
                col, row = chess.square_file(
                    move.to_square), 7 - chess.square_rank(move.to_square)
                pygame.draw.circle(screen, (0, 255, 0),
                                   (col * SQUARE_SIZE + SQUARE_SIZE // 2,
                                    row * SQUARE_SIZE + SQUARE_SIZE // 2),
                                   SQUARE_SIZE // 4)

    def get_possible_moves(self):
        """Returns a list of legal moves for the selected piece."""
        if self.selected_square:
            row, col = self.selected_square
            square = chess.square(col, 7 - row)
            piece = self.board.piece_at(square)
            if piece is not None:
                legal_moves = self.board.legal_moves
                return [move for move in legal_moves if move.from_square == square]
        return []

    def draw_game_over(self, screen):
        """Draws a semi-transparent overlay with the game over message and a restart button."""
        # Create a semi-transparent surface to overlay on the board
        overlay = pygame.Surface(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((200, 200, 200, 128))  # Fill with gray, half transparent

        font = pygame.font.Font(None, 36)
        if self.board.is_checkmate():
            text = "Checkmate! " + ("Black" if self.board.turn else "White") + " wins!"
        elif self.board.is_stalemate():
            text = "Draw by Stalemate!"
        else:
            text = "Game Over"

        text_surface = font.render(text, True, (0, 0, 0))
        text_rect = text_surface.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50))
        overlay.blit(text_surface, text_rect)

        # Draw the Restart button on the overlay
        draw_button(overlay, "Restart", SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2 + 50, 100, 50,
                    (255, 255, 255), (200, 200, 200), self.reset)

        # Blit (draw) the overlay onto the screen
        screen.blit(overlay, (0, 0))

# --- Helper Function ---
def draw_button(screen, text, x, y, width, height, color, hover_color, action=None):
    """Draws a button and handles click events."""
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x + width > mouse[0] > x and y + height > mouse[1] > y:
        pygame.draw.rect(screen, hover_color, (x, y, width, height))
        if click[0] == 1 and action is not None:
            action()
    else:
        pygame.draw.rect(screen, color, (x, y, width, height))

    font = pygame.font.Font(None, 20)
    text_surface = font.render(text, True, (0, 0, 0))
    text_rect = text_surface.get_rect(
        center=((x + (width / 2)), (y + (height / 2))))
    screen.blit(text_surface, text_rect)

# --- Main Game Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Shojin Chess")

    # Replace with your engine path
    engine_path = "C:/Users/leonl/Shojin/AdaptedChess/python_chess_stockfish/engine/stockfish-windows-x86-64-avx2.exe"
    game = Game(engine_path)

    running = True
    while running:
        game.make_ai_move()  # AI's turn
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    game.handle_mouse_click(pygame.mouse.get_pos())
                elif event.button == 3:  # Right click
                    game.evolution_mode = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_j:  # Press 'j' for jin mode
                    game.jin_mode = True

        game.update_game_state()
        game.draw(screen)
        pygame.display.flip()

    game.engine.quit()
    pygame.quit()

if __name__ == "__main__":
    main()
