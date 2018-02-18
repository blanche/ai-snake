import curses
import math

from SnakeGame import SnakeGame

import sys

sys.path.append('/opt/lukas/pycharm-2017.3/debug-eggs/pycharm-debug.egg')
import pydevd

pydevd.settrace('localhost', port=5000, stdoutToServer=True, stderrToServer=True, suspend=False)

BOARD_WIDTH = 60
BOARD_HEIGHT = 20
g = SnakeGame(BOARD_WIDTH, BOARD_HEIGHT)

curses.initscr()
win = curses.newwin(BOARD_HEIGHT + 10, BOARD_WIDTH, 0, 0)
win.keypad(1)
curses.noecho()
curses.curs_set(0)
win.border(0)
win.nodelay(1)

win.addch(g.food[0], g.food[1], '*')  # Prints the food
key = curses.KEY_RIGHT


def display_bar():
    for i, val in enumerate(g.get_distance_to_food()):
        win.addstr(BOARD_HEIGHT + i + 1, 1, "{}: {}    ".format(i, val))


while key != 27:  # While Esc key is not pressed
    win.border(0)
    win.addstr(0, 2, 'Score : ' + str(g.score) + ' ')  # Printing 'Score' and
    win.addstr(0, 27, ' SNAKE ')  # 'SNAKE' strings

    # Increases the speed of Snake as its length increases
    # win.timeout(int(math.floor(150 - (len(g.snake) / 5 + len(g.snake) / 10) % 120)))
    win.timeout(100000)

    prevKey = key  # Previous key pressed
    event = win.getch()
    key = key if event == -1 else event

    last_score = g.score
    reward, done = g.step(key)
    display_bar()
    if not done:
        if g.score > last_score:  # snake has eaten
            win.addch(g.food[0], g.food[1], '*')
        else:
            win.addch(g.last_pop[0], g.last_pop[1], ' ')

        win.addch(g.snake[0][0], g.snake[0][1], '#')
    else:
        break

curses.endwin()
print("\nScore - " + str(g.score))
