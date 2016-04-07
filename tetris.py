import pygame, sys, random
from pygame.locals import *
from random import choice

class Piece:
    O = (((0,0,0,0,0), (0,0,0,0,0),(0,0,1,1,0),(0,0,1,1,0),(0,0,0,0,0)),) * 4

    I = (((0,0,0,0,0),(0,0,0,0,0),(0,1,1,1,1),(0,0,0,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,1,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(1,1,1,1,0),(0,0,0,0,0),(0,0,0,0,0)),
         ((0,0,1,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,0,0,0)))

    L = (((0,0,0,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,1,1,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(0,1,1,1,0),(0,1,0,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,1,1,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,1,0),(0,1,1,1,0),(0,0,0,0,0),(0,0,0,0,0)))

    J = (((0,0,0,0,0),(0,0,1,0,0),(0,0,1,0,0),(0,1,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,1,0,0,0),(0,1,1,1,0),(0,0,0,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,1,0),(0,0,1,0,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(0,1,1,1,0),(0,0,0,1,0),(0,0,0,0,0)))

    Z = (((0,0,0,0,0),(0,0,0,1,0),(0,0,1,1,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(0,1,1,0,0),(0,0,1,1,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,0,0),(0,1,1,0,0),(0,1,0,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,1,1,0,0),(0,0,1,1,0),(0,0,0,0,0),(0,0,0,0,0)))

    S = (((0,0,0,0,0),(0,0,1,0,0),(0,0,1,1,0),(0,0,0,1,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(0,0,1,1,0),(0,1,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,1,0,0,0),(0,1,1,0,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,1,0),(0,1,1,0,0),(0,0,0,0,0),(0,0,0,0,0)))

    T = (((0,0,0,0,0),(0,0,1,0,0),(0,0,1,1,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,0,0,0),(0,1,1,1,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,0,0),(0,1,1,0,0),(0,0,1,0,0),(0,0,0,0,0)),
         ((0,0,0,0,0),(0,0,1,0,0),(0,1,1,1,0),(0,0,0,0,0),(0,0,0,0,0)))

    PIECES = {'O': O, 'I': I, 'L': L, 'J': J, 'Z': Z, 'S':S, 'T':T}

    def __init__(self, piece_name=None):
        if piece_name:
            self.piece_name = piece_name
        else:
            self.piece_name = random.choice(Piece.PIECES.keys())
        self.rotation = 0
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]

    def __iter__(self):
        for row in self.array2d:
            yield row

    def rotate(self, clockwise=True):
        self.rotation = (self.rotation + 1) % 4 if clockwise else \
                        (self.rotation - 1) % 4
        self.array2d = Piece.PIECES[self.piece_name][self.rotation]

class Board:
    COLLIDE_ERROR = {'no_error': 0, 'right_wall': 1, 'left_wall': 2,
                     'bottom': 3, 'overlap': 4}

    def __init__(self, screen):
        pygame.font.init()
        self.score = 0
        self.screen = screen
        self.width = 10
        self.height = 22
        self.block_size = 25
        self.board = []
        for _ in xrange(self.height):
            self.board.append([0] * self.width)
        self.generate_piece()
        self.set_next_piece()

        self.droped = False

    def generate_piece(self, piece_name = None):
        self.piece = Piece(piece_name)
        self.piece_x, self.piece_y = 3, 0

    def absorb_piece(self):
        for y, row in enumerate(self.piece):
            for x, block in enumerate(row):
                if block:
                    self.board[y+self.piece_y][x+self.piece_x] = block

        self.generate_piece(self.next_piece)
        self.set_next_piece()

    def _block_collide_with_board(self, x, y):
        if x < 0: 
            return Board.COLLIDE_ERROR['left_wall']
        elif x >= self.width:
            return Board.COLLIDE_ERROR['right_wall']
        elif y >= self.height:
            return Board.COLLIDE_ERROR['bottom']
        elif self.board[y][x]:
            return Board.COLLIDE_ERROR['overlap']
        return Board.COLLIDE_ERROR['no_error'] 

    def collide_with_board(self, dx, dy):
        """Check if piece (offset dx, dy) collides with board"""
        for y, row in enumerate(self.piece):
            for x, block in enumerate(row):
                if block:
                    collide = self._block_collide_with_board(x=x+dx, y=y+dy)
                    if collide:
                        return collide
        return Board.COLLIDE_ERROR['no_error']

    def get_stack_height(self):
        stack_height = 0
        for i in range(0, self.height):
            blank_row = True
            for j in range(0, self.width):
                if self.board[i][j] != 0:
                    blank_row = False
            if not blank_row:
                stack_height = self.height - i
                break
        return stack_height

    def get_blank_blocks(self):
        blank_blocks = 0
        for i in range(0, self.height):
            for j in range(0, self.width):
                if self.board[i][j] != 0:
                    blank_blocks += 1

        return blank_blocks

    def set_next_piece(self):
        self.next_piece = choice(['O', 'I', 'L', 'J', 'Z', 'S', 'T'])

    def _can_move_piece(self, dx, dy):
        dx_ = self.piece_x + dx
        dy_ = self.piece_y + dy
        if self.collide_with_board(dx=dx_, dy=dy_):
            return False
        return True

    def _can_drop_piece(self):
        return self._can_move_piece(dx=0, dy=1)

    def _try_rotate_piece(self, clockwise=True):
        self.piece.rotate(clockwise)
        collide = self.collide_with_board(dx=self.piece_x, dy=self.piece_y)
        if not collide:
            pass
        elif collide == Board.COLLIDE_ERROR['left_wall']:
            if self._can_move_piece(dx=1, dy=0):
                self.move_piece(dx=1, dy=0)
            elif self._can_move_piece(dx=2, dy=0):
                self.move_piece(dx=2, dy=0)
            else:
                self.piece.rotate(not clockwise)
        elif collide == Board.COLLIDE_ERROR['right_wall']:
            if self._can_move_piece(dx=-1, dy=0):
                self.move_piece(dx=-1, dy=0)
            elif self._can_move_piece(dx=-2, dy=0):
                self.move_piece(dx=-2, dy=0)
            else:
                self.piece.rotate(not clockwise)
        else:
            self.piece.rotate(not clockwise)

    def move_piece(self, dx, dy):
        if self._can_move_piece(dx, dy):
            self.piece_x += dx
            self.piece_y += dy

    def drop_piece(self):
        if self._can_drop_piece():
            self.move_piece(dx=0, dy=1)
            self.droped = False
        else:
            self.absorb_piece()
            self.delete_lines()
            self.droped = True

    def full_drop_piece(self):
        while self._can_drop_piece():
            self.drop_piece()
        self.drop_piece()

    def rotate_piece(self, clockwise=True):
        self._try_rotate_piece(clockwise)

    def pos_to_pixel(self, x, y):
        return self.block_size*x, self.block_size*(y-2)

    def _delete_line(self, y):
        for y in reversed(xrange(1, y+1)):
            self.board[y] = list(self.board[y-1])

    def delete_lines(self):
        remove = [y for y, row in enumerate(self.board) if all(row)]
        count = 0
        for y in remove:
            count += 1
            self.score += 1000 * count
            self._delete_line(y)    

    def game_over(self):
        return sum(self.board[0]) > 0 or sum(self.board[1]) > 0

    def draw_blocks(self, array2d, color=(255,255,255), dx=0, dy=0):
        for y, row in enumerate(array2d):
            y += dy
            if y >= 2 and y < self.height:
                for x, block in enumerate(row):
                    if block:
                        x += dx
                        x_pix, y_pix = self.pos_to_pixel(x, y)
                        # draw block
                        pygame.draw.rect(self.screen, color,
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size))
                        # draw border
                        pygame.draw.rect(self.screen, (0, 0, 0),
                                         (  x_pix, y_pix,
                                            self.block_size,
                                            self.block_size), 1)

    def draw_ui(self):
        pygame.draw.rect(self.screen, (255, 255, 255),
                            (250, 0, 2, 500), 1)

        font = pygame.font.Font(None, 30)
        score_text = font.render("SCORE: " + str(self.score), 1, (255,255,0))
        self.screen.blit(score_text, (270, 10))

        next_block_text = font.render("Next block", 1, (255,255,255))
        self.screen.blit(next_block_text, (270, 50))

        self.draw_blocks(Piece(self.next_piece), dx=10, dy=5)
    
    def draw(self):
        self.draw_ui()
        self.draw_blocks(self.piece, dx=self.piece_x, dy=self.piece_y)
        self.draw_blocks(self.board)

class Tetris:
    DROP_EVENT = USEREVENT + 1

    def __init__(self):
        self.screen = pygame.display.set_mode((400, 500))
        self.clock = pygame.time.Clock()
        self.board = Board(self.screen)
        pygame.init()
        pygame.time.set_timer(self.DROP_EVENT, 500)

        self.saved_height = 0

    def handle_key(self, event_key):
        if event_key == K_DOWN or event_key == 0:
            self.board.drop_piece()
        elif event_key == K_LEFT or event_key == 1:
            self.board.move_piece(dx=-1, dy=0)
        elif event_key == K_RIGHT or event_key == 2:
            self.board.move_piece(dx=1, dy=0)
        elif event_key == K_UP or event_key == 3:
            self.board.rotate_piece()
        elif event_key == K_SPACE or event_key == 4:
            self.board.full_drop_piece()
        elif event_key == K_ESCAPE:
            self.pause()

    def pause(self):
        running = True 
        while running:
            for event in pygame.event.get():
                if event.type == KEYDOWN and event.key == K_ESCAPE:
                    running = False

    def run(self):
        while True:
            old_score = self.board.score
            if self.board.game_over():
                print "Game over and restart"
                self.board = Board(self.screen)
                #pygame.quit()
                #sys.exit()
            self.screen.fill((0, 0, 0))
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == Tetris.DROP_EVENT:
                    self.board.drop_piece()
            
            self.board.draw()
            pygame.display.update()
            self.clock.tick(60)

            if self.board._can_drop_piece():
                print 1

    def step(self, key_event):
        old_score = self.board.score
        reward = 0
        gameover = False

        self.screen.fill((0, 0, 0))
        self.handle_key(key_event)

        self.board.draw()

        pygame.display.update()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == self.DROP_EVENT:
                self.board.drop_piece()

        image_data = pygame.surfarray.array3d(pygame.display.get_surface())

        if self.board.droped == True:
            reward = self.saved_height - self.board.get_stack_height()
            self.saved_height = self.board.get_stack_height()

        if self.board.game_over():
            print "Game over and restart"
            self.board = Board(self.screen)
            reward = -200
            gameover = True
            return image_data, reward, gameover

        if old_score < self.board.score:
            reward = self.board.score - old_score

        return image_data, reward, gameover

if __name__ == "__main__":
    Tetris().run()
