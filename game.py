import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.font.init()

Point = namedtuple("Point", "x, y")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
FT_BLUE = (0, 186, 188)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


BLOCK_SIZE = 20

GREEN_APPLES = 2
RED_APPLES = 1


class SnakeGame:
    def __init__(self, width=760, height=520):
        if width % (BLOCK_SIZE * 2) != 0 or height % (BLOCK_SIZE * 2) != 0:
            raise Exception(
                "Width and Height must be multiples of " + str(BLOCK_SIZE * 2)
            )

        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Learn2Slither")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = random.choice(list(Direction))
        self.head = Point(
            random.randint(0, (self.width - BLOCK_SIZE)
                           // BLOCK_SIZE) * BLOCK_SIZE,
            random.randint(0, (self.height - BLOCK_SIZE)
                           // BLOCK_SIZE) * BLOCK_SIZE,
        )
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.green_apples = []
        self.red_apples = []
        self._place_food()

    def _place_food(self):
        while len(self.green_apples) < GREEN_APPLES:
            x = random.randint(0, (self.width - BLOCK_SIZE)
                               // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE)
                               // BLOCK_SIZE) * BLOCK_SIZE

            if (
                Point(x, y) in self.snake
                or Point(x, y) in self.green_apples
                or Point(x, y) in self.red_apples
            ):
                continue

            self.green_apples.append(Point(x, y))

        while len(self.red_apples) < RED_APPLES:
            x = random.randint(0, (self.width - BLOCK_SIZE)
                               // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE)
                               // BLOCK_SIZE) * BLOCK_SIZE

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
        fps: int = 0,
        to_display: list[str] = []
    ):
        self.direction = self._move(
            direction if direction is not None else self.direction
        )

        self.snake.insert(0, self.head)
        self.score = len(self.snake) - 3

        game_over = False
        reward = -1

        if self.head in self.green_apples:
            reward = 5
            self.green_apples.remove(self.head)
            self._place_food()
        elif self.head in self.red_apples:
            reward = -5
            self.red_apples.remove(self.head)
            self._place_food()
            self.snake.pop()
            self.snake.pop()
        else:
            self.snake.pop()

        self._update_ui(to_display)
        self.clock.tick(fps)

        if len(self.snake) <= 0 or self.is_collision(self.head):
            reward = -100
            game_over = True

        return reward, game_over, self.score

    def is_collision(self, point):
        if (
            point.x > self.width - BLOCK_SIZE
            or point.x < 0
            or point.y > self.height - BLOCK_SIZE
            or point.y < 0
        ):
            return True
        if point in self.snake[1:]:
            return True

        return False

    def _update_ui(self, to_display=[]):
        self.display.fill(WHITE)

        for pt in self.snake:
            pygame.draw.rect(
                self.display, FT_BLUE,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )

        for pt in self.green_apples:
            pygame.draw.rect(
                self.display, GREEN,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )

        for pt in self.red_apples:
            pygame.draw.rect(
                self.display, RED,
                pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            )

        text = "Score: " + str(self.score)
        font = pygame.font.Font(None, 30)
        score = font.render(text, True, FT_BLUE)
        x = BLOCK_SIZE
        y = BLOCK_SIZE
        for line in to_display:
            score = font.render(line, True, FT_BLUE)
            self.display.blit(score, (x, y))
            y += BLOCK_SIZE * 1.5

        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

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

    def get_state(self):
        head = self.snake[0] if len(self.snake) > 0 else self.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        clock = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current = clock.index(self.direction)
        straight = clock[current]
        left = clock[(current - 1) % 4]
        right = clock[(current + 1) % 4]

        state = [
            {
                "label": "danger_straight",
                "value": (dir_r and self.is_collision(point_r))
                or (dir_l and self.is_collision(point_l))
                or (dir_u and self.is_collision(point_u))
                or (dir_d and self.is_collision(point_d)),
            },
            {
                "label": "danger_left",
                "value": (dir_d and self.is_collision(point_r))
                or (dir_u and self.is_collision(point_l))
                or (dir_r and self.is_collision(point_u))
                or (dir_l and self.is_collision(point_d)),
            },
            {
                "label": "danger_right",
                "value": (dir_u and self.is_collision(point_r))
                or (dir_d and self.is_collision(point_l))
                or (dir_l and self.is_collision(point_u))
                or (dir_r and self.is_collision(point_d)),
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

    # def get_state(self):
    #     head = self.snake[0]

    #     left_wall = Point(0, head.y)
    #     right_wall = Point(self.width - BLOCK_SIZE, head.y)
    #     top_wall = Point(head.x, 0)
    #     bottom_wall = Point(head.x, self.height - BLOCK_SIZE)

    #     green_apple_up = \
    #         Direction.UP == self._direction(head, self.green_apples[0])\
    #         or Direction.UP == self._direction(head, self.green_apples[1])

    #     green_apple_down = \
    #         Direction.DOWN == self._direction(head, self.green_apples[0])\
    #         or Direction.DOWN == self._direction(head, self.green_apples[1])

    #     green_apple_left = \
    #         Direction.LEFT == self._direction(head, self.green_apples[0])\
    #         or Direction.LEFT == self._direction(head, self.green_apples[1])

    #     green_apple_right = \
    #         Direction.RIGHT == self._direction(head, self.green_apples[0])\
    #         or Direction.RIGHT == self._direction(head, self.green_apples[1])

    #     red_apple_up = \
    #         Direction.UP == self._direction(head, self.red_apples[0])
    #     red_apple_down = \
    #         Direction.DOWN == self._direction(head, self.red_apples[0])
    #     red_apple_left = \
    #         Direction.LEFT == self._direction(head, self.red_apples[0])
    #     red_apple_right = \
    #         Direction.RIGHT == self._direction(head, self.red_apples[0])

    # state = [
    #     {"label": "move_up", "value": self.direction == Direction.UP},
    #     {"label": "move_down", "value": self.direction == Direction.DOWN},
    #     {"label": "move_left", "value": self.direction == Direction.LEFT},
    #     {"label": "move_right", "value": self.direction == Direction.RIGHT},

    #     {"label": "distance_top_wall",
    #      "value": self._distance(head, top_wall)},
    #     {"label": "distance_bottom_wall",
    #      "value": self._distance(head, bottom_wall)},
    #     {"label": "distance_left_wall",
    #      "value": self._distance(head, left_wall)},
    #     {"label": "distance_right_wall",
    #      "value": self._distance(head, right_wall)},

    #     {"label": "distance_green_apple_1",
    #      "value": self._distance(head, self.green_apples[0])},
    #     {"label": "distance_green_apple_2",
    #      "value": self._distance(head, self.green_apples[1])},
    #     {"label": "distance_red_apple",
    #      "value": self._distance(head, self.red_apples[0])},

    #     {"label": "green_apple_up", "value": green_apple_up},
    #     {"label": "green_apple_down", "value": green_apple_down},
    #     {"label": "green_apple_left", "value": green_apple_left},
    #     {"label": "green_apple_right", "value": green_apple_right},

    #     {"label": "red_apple_up", "value": red_apple_up},
    #     {"label": "red_apple_down", "value": red_apple_down},
    #     {"label": "red_apple_left", "value": red_apple_left},
    #     {"label": "red_apple_right", "value": red_apple_right},
    # ]

    #     return state


# if __name__ == '__main__':
#     game = SnakeGame()

#     while True:
#         action = None
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 pygame.quit()
#                 quit()
#             elif event.type == pygame.KEYDOWN:
#                 if event.key == pygame.K_UP:
#                     action = Direction.UP
#                 elif event.key == pygame.K_DOWN:
#                     action = Direction.DOWN
#                 elif event.key == pygame.K_LEFT:
#                     action = Direction.LEFT
#                 elif event.key == pygame.K_RIGHT:
#                     action = Direction.RIGHT

#         reward, game_over, score = game.play_step(action)

#         if game_over:
#             break

#     print('Final Score', score)

#     pygame.quit()
