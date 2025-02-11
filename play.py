from agent import Agent
from game import SnakeGame, Direction
import pygame
from plot import plot
import sys


def main():
    step_by_step = any(arg == '--step' for arg in sys.argv)
    manual = any(arg == '--manual' for arg in sys.argv)

    best_score = 0
    games = 0
    scores = []
    mean_scores = []

    agent = Agent(
        gamma=0.9,
        epsilon=0.1,
        lr=0.001,
        max_memory=100_000,
        batch_size=1_000
    )
    game = SnakeGame()

    while True:
        move = None
        execute = not step_by_step

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                break
            elif event.type == pygame.KEYDOWN:
                execute = True
                if event.key == pygame.K_SPACE:
                    execute = not execute
                if event.key == pygame.K_LEFT:
                    move = Direction.LEFT
                elif event.key == pygame.K_UP:
                    move = Direction.UP
                elif event.key == pygame.K_RIGHT:
                    move = Direction.RIGHT
                elif event.key == pygame.K_DOWN:
                    move = Direction.DOWN

        if not execute:
            continue

        state = game.get_state()

        if not manual:
            action = agent.get_action(state)
            move = game.relative_to_absolute(action)

        (reward, done, score) = game.play_step(direction=move)

        next_state = game.get_state()

        logs = [
            f"Score: {score}",
            f"Highest Score: {best_score}",
            f"Games Played: {games}",
            f"Head: {game.snake[0] if len(game.snake) > 0 else None}",
            f"Move: {move}",
            f"State: {state}",
            f"Next State: {next_state}",
            f"Reward: {reward}",
            f"Done: {done}",
        ]

        for log in logs:
            print(log)

        print()

        if not manual:
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)
            if done:
                agent.train_long_memory()

        if done:
            game.reset()
            games += 1

            if score > best_score:
                best_score = score
                agent.model.save("best_model" + str(score) + ".pth")

            scores.append(score)
            mean_scores.append(sum(scores) / max(games, 1))
            plot(scores, mean_scores)


if __name__ == "__main__":
    main()
