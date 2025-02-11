from agent import Agent
from game import SnakeGame, Direction
import pygame
from plot import plot
import sys
from settings import config


def main():
    step_by_step = any(arg == '--step' for arg in sys.argv)
    manual = any(arg == '--manual' for arg in sys.argv)
    train = any(arg == '--train' for arg in sys.argv) and not manual
    fps = 60 if not manual else 0
    load_model = \
        next((arg for arg in sys.argv if arg.startswith('--model=')), None) \
        .split('=')[1] \
        if not manual else None

    best_score = 0
    games = 0
    scores = []
    mean_scores = []

    agent = Agent(
        gamma=config['gamma'],
        epsilon_init=config['epsilon_init'] if train else 0,
        lr=config['lr'],
        max_memory=config['max_memory'],
        batch_size=config['batch_size'],
    )

    if load_model is not None:
        agent.model.load(config['models_path'] + load_model)

    game = SnakeGame(
        width=config['game_width'],
        height=config['game_height'],
        fps=fps,
        green_apples_count=config['green_apples_count'],
        red_apples_count=config['red_apples_count'],
    )

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

        state_with_labels = game.get_state()
        state = [element['value'] for element in state_with_labels]

        if not manual:
            action = agent.get_action(state)
            move = game.relative_to_absolute(action)

        (reward, done, score) = game.play_step(
            direction=move,
        )

        next_state_with_labels = game.get_state()
        next_state = [element['value'] for element in next_state_with_labels]

        logs = [
            f"Score: {score}",
            f"Highest Score: {best_score}",
            f"Games Played: {games}",
            f"Head: {game.snake[0] if len(game.snake) > 0 else None}",
            f"Move: {move}",
            f"Reward: {reward}",
            f"Done: {done}",
        ]

        for element in state_with_labels:
            logs.append(f"State {element['label']}: {element['value']}")

        for element in next_state_with_labels:
            logs.append(f"Next State {element['label']}: {element['value']}")

        for log in logs:
            print(log)

        print()

        if train:
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

        if done:
            game.reset()
            games += 1

            if train:
                agent.train_long_memory()
                agent.epsilon = max(
                    config['epsilon_min'],
                    agent.epsilon * config['epsilon_decay']
                )

                scores.append(score)
                mean_scores.append(sum(scores) / max(games, 1))
                plot(scores, mean_scores)

            if score > best_score:
                best_score = score
                if train:
                    agent.model.save(
                        config['models_path'] + "best_model"
                        + str(score) + ".pth"
                    )


if __name__ == "__main__":
    main()
