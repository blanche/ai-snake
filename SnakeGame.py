from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint


class SnakeGame:
    def __init__(self, board_width, board_height):
        self.BOARD_WIDTH = board_width
        self.BOARD_HEIGHT = board_height

        self.key = KEY_RIGHT
        self.prev_key = self.key
        self.score = 0

        # Initial snake co-ordinates
        self.snake = [[4, 10], [4, 9], [4, 8]]
        # First food co-ordinates
        self.food = [10, 20]
        self.last_pop = [0, 0]

    def step(self, new_key):
        self.prev_key = self.key
        self.key = new_key

        # If an invalid key is pressed or opposite direction
        if self.key not in [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN, 27] \
                or self.key == KEY_LEFT and self.prev_key == KEY_RIGHT \
                or self.key == KEY_RIGHT and self.prev_key == KEY_LEFT \
                or self.key == KEY_UP and self.prev_key == KEY_DOWN \
                or self.key == KEY_DOWN and self.prev_key == KEY_UP:
            self.key = self.prev_key

        self.snake.insert(0, [self.snake[0][0] + (self.key == KEY_DOWN and 1) + (self.key == KEY_UP and -1),
                              self.snake[0][1] + (self.key == KEY_LEFT and -1) + (self.key == KEY_RIGHT and 1)])

        # If snake crosses the boundaries, make it enter from the other side
        if self.snake[0][0] == 0: self.snake[0][0] = self.BOARD_HEIGHT - 2
        if self.snake[0][1] == 0: self.snake[0][1] = self.BOARD_WIDTH - 2
        if self.snake[0][0] == self.BOARD_HEIGHT - 1: self.snake[0][0] = 1
        if self.snake[0][1] == self.BOARD_WIDTH - 1: self.snake[0][1] = 1

        # Exit if snake crosses the boundaries (Uncomment to enable)
        # if self.snake[0][0] == 0 or self.snake[0][0] == self.BOARD_HEIGHT - 1 \
        #         or self.snake[0][1] == 0 or self.snake[0][1] == self.BOARD_WIDTH - 1:
        #     return 10

        # If snake runs over itself
        if self.snake[0] in self.snake[1:]:
            return 1

        if self.snake[0] == self.food:  # When snake eats the food
            new_food = []
            self.score += 1
            while not new_food:
                new_food = [randint(1, 18), randint(1, 58)]  # Calculating next food's coordinates
                if new_food in self.snake: new_food = []
            self.food = new_food
        else:
            # [1] If it does not eat the food, length decreases
            self.last_pop = self.snake.pop()

        return 0
