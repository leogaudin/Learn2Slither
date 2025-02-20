import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()

Point = namedtuple("Point", "x, y")

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
FT_RED = (232, 33, 39)
FT_BLUE = (0, 186, 188)
FT_GREEN = (15, 218, 83)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class SnakeGame:
    def __init__(
        self,
        width=760,
        height=520,
        block_size=20,
        fps=0,
        green_apple_reward=20,
        red_apple_reward=-20,
        alive_reward=-0.5,
        death_reward=-100,
        invisible=False,
    ):
        if width % (block_size * 2) != 0 or height % (block_size * 2) != 0:
            raise Exception(
                "Width and Height must be multiples of " + str(block_size * 2)
            )

        self.width = width
        self.height = height
        self.block_size = block_size
        self.font_size = height // 30
        self.font = pygame.font.Font('assets/cmunrm.ttf',
                                     self.font_size)
        self.fps = fps
        self.green_apples_count = np.mean([width, height]) \
            // block_size // 10 * 2
        self.red_apples_count = self.green_apples_count // 2
        self.green_apple_reward = green_apple_reward
        self.red_apple_reward = red_apple_reward
        self.alive_reward = alive_reward
        self.death_reward = death_reward
        self.invisible = invisible

        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Learn2Slither")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.move_history = []
        self.time_alive = 0
        self.direction = random.choice(list(Direction))
        self.head = Point(
            random.randint(0, (self.width - self.block_size)
                           // self.block_size) * self.block_size,
            random.randint(0, (self.height - self.block_size)
                           // self.block_size) * self.block_size,
        )
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y),
        ]
        self.score = 0
        self.green_apples = []
        self.red_apples = []
        self._place_food()

    def _place_food(self):
        while len(self.green_apples) < self.green_apples_count:
            x = random.randint(0, (self.width - self.block_size)
                               // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size)
                               // self.block_size) * self.block_size

            if (
                Point(x, y) in self.snake
                or Point(x, y) in self.green_apples
                or Point(x, y) in self.red_apples
            ):
                continue

            self.green_apples.append(Point(x, y))

        while len(self.red_apples) < self.red_apples_count:
            x = random.randint(0, (self.width - self.block_size)
                               // self.block_size) * self.block_size
            y = random.randint(0, (self.height - self.block_size)
                               // self.block_size) * self.block_size

            if (
                Point(x, y) in self.snake
                or Point(x, y) in self.green_apples
                or Point(x, y) in self.red_apples
            ):
                continue

            self.red_apples.append(Point(x, y))

    def play_step(
        self,
        direction=None,
        to_display: list[str] = []
    ):
        self.time_alive += 1
        self.direction = self._move(
            direction if direction is not None else self.direction
        )
        self.move_history.append({
            "head": self.head,
            "move": self.direction
        })

        self.snake.insert(0, self.head)
        self.score = len(self.snake) - 3

        game_over = False
        reward = self.alive_reward / (self._move_index() ** 3) \
            if len(self.move_history) > 2 \
            else self.alive_reward

        if self.head in self.green_apples:
            reward = self.green_apple_reward
            self.green_apples.remove(self.head)
            self._place_food()
        elif self.head in self.red_apples:
            reward = self.red_apple_reward
            self.red_apples.remove(self.head)
            self._place_food()
            self.snake.pop()
            self.snake.pop()
        else:
            self.snake.pop()

        if not self.invisible:
            self._update_ui(to_display)
            self.clock.tick(self.fps)

        if len(self.snake) <= 0 \
                or self.is_collision(self.head):
            reward = self.death_reward
            game_over = True

        return reward, game_over, self.score

    def is_collision(self, point):
        if (
            point.x > self.width - self.block_size
            or point.x < 0
            or point.y > self.height - self.block_size
            or point.y < 0
        ):
            return True
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self, to_display=[]):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, WHITE,
                pygame.Rect(pt.x, pt.y, self.block_size, self.block_size)
            )

        for pt in self.green_apples:
            pygame.draw.circle(
                self.display, FT_GREEN,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        for pt in self.red_apples:
            pygame.draw.circle(
                self.display, RED,
                (pt.x + self.block_size // 2, pt.y + self.block_size // 2),
                self.block_size // 2
            )

        x, y = self.block_size // 2, self.block_size // 2
        for line in to_display:
            line = self.font.render(line, True, WHITE)
            self.display.blit(line, (x, y))
            y += self.font_size * 1.5

        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size

        self.head = Point(x, y)

        return direction

    def _distance(self, point1, point2):
        if not (point1.x == point2.x or point1.y == point2.y):
            return -1

        distance = ((point1.x - point2.x) ** 2
                    + (point1.y - point2.y) ** 2) ** 0.5

        return distance

    def _direction(self, point1, point2):
        direction = None
        if point1.x == point2.x:
            if point1.y < point2.y:
                direction = Direction.DOWN
            else:
                direction = Direction.UP
        elif point1.y == point2.y:
            if point1.x < point2.x:
                direction = Direction.RIGHT
            else:
                direction = Direction.LEFT

        return direction

    def relative_to_absolute(self, direction):
        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        i = clock.index(self.direction)
        choices = [
            i,  # straight
            (i - 1) % 4,  # left
            (i + 1) % 4,  # right
        ]
        return clock[choices[np.argmax(direction)]]

    def _is_there_point(self, from_point, to_points, direction):
        if direction == Direction.RIGHT:
            return any([from_point.x < to_point.x
                        and from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.LEFT:
            return any([from_point.x > to_point.x
                        and from_point.y == to_point.y
                        for to_point in to_points])
        elif direction == Direction.UP:
            return any([from_point.y > to_point.y
                        and from_point.x == to_point.x
                        for to_point in to_points])
        elif direction == Direction.DOWN:
            return any([from_point.y < to_point.y
                        and from_point.x == to_point.x
                        for to_point in to_points])

        return False

    def _move_index(self):
        if len(self.move_history) < 1:
            return 1

        coordinates = [move['head'] for move in self.move_history[-10:]]
        x = np.array([point.x for point in coordinates])
        y = np.array([point.y for point in coordinates])
        std_dev = np.mean([
            np.std(x),
            np.std(y)
        ]) / self.block_size

        return std_dev

    def get_state(self):
        head = self.snake[0] if len(self.snake) > 0 else self.head
        direct_left = Point(head.x - self.block_size, head.y)
        direct_right = Point(head.x + self.block_size, head.y)
        direct_up = Point(head.x, head.y - self.block_size)
        direct_down = Point(head.x, head.y + self.block_size)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        current = clock.index(self.direction)
        straight = clock[current]
        left = clock[(current - 1) % 4]
        right = clock[(current + 1) % 4]

        previous = self.move_history[-2]['move'] if len(
            self.move_history) > 1 else self.direction
        previous = clock.index(previous)
        last_move_straight = previous == current
        last_move_left = previous == (current - 1) % 4
        last_move_right = previous == (current + 1) % 4

        state = [
            {
                "label": "move_index",
                "value": self._move_index(),
            },
            {
                "label": "last_move_straight",
                "value": last_move_straight,
            },
            {
                "label": "last_move_left",
                "value": last_move_left,
            },
            {
                "label": "last_move_right",
                "value": last_move_right,
            },
            {
                "label": "danger_straight",
                "value": (dir_r and self.is_collision(direct_right))
                or (dir_l and self.is_collision(direct_left))
                or (dir_u and self.is_collision(direct_up))
                or (dir_d and self.is_collision(direct_down)),
            },
            {
                "label": "danger_left",
                "value": (dir_d and self.is_collision(direct_right))
                or (dir_u and self.is_collision(direct_left))
                or (dir_r and self.is_collision(direct_up))
                or (dir_l and self.is_collision(direct_down)),
            },
            {
                "label": "danger_right",
                "value": (dir_u and self.is_collision(direct_right))
                or (dir_d and self.is_collision(direct_left))
                or (dir_l and self.is_collision(direct_up))
                or (dir_r and self.is_collision(direct_down)),
            },
            {"label": "green_apple_straight",
             "value": self._is_there_point(head, self.green_apples, straight)},
            {"label": "green_apple_left",
             "value": self._is_there_point(head, self.green_apples, left)},
            {"label": "green_apple_right",
             "value": self._is_there_point(head, self.green_apples, right)},
            {"label": "red_apple_straight",
             "value": self._is_there_point(head, self.red_apples, straight)},
            {"label": "red_apple_left",
             "value": self._is_there_point(head, self.red_apples, left)},
            {"label": "red_apple_right",
             "value": self._is_there_point(head, self.red_apples, right)}
        ]

        return state
