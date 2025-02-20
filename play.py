from agent import Agent
from game import SnakeGame, Direction
import pygame
from plot import plot
import sys
from settings import config, get_args


def main():
    (
        step_by_step,
        manual,
        train,
        fps,
        load_model,
        episodes,
        invisible,
        verbose,
    ) = get_args(sys.argv)

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
        block_size=config['block_size'],
        fps=fps,
        alive_reward=config['alive_reward'],
        death_reward=config['death_reward'],
        green_apple_reward=config['green_apple_reward'],
        red_apple_reward=config['red_apple_reward'],
        invisible=invisible,
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

        prev_state_with_labels = game.get_state()
        prev_state = [element['value'] for element in prev_state_with_labels]

        if not manual:
            action = agent.get_action(prev_state)
            move = game.relative_to_absolute(action)

        (reward, done, score) = game.play_step(
            direction=move,
        )

        next_state_with_labels = game.get_state()
        next_state = [element['value'] for element in next_state_with_labels]

        if verbose:
            logs = [
                f"Score: {score}",
                f"Mean Score: {mean_scores[-1] if mean_scores else 0}",
                f"Highest Score: {best_score}",
                f"Games Played: {games}",
                f"Head: {game.snake[0] if len(game.snake) > 0 else None}",
                f"Direction: {move}",
                f"Reward: {reward}",
                f"Done: {done}",
            ]

            for element in prev_state_with_labels:
                logs.append(f"Previous State {element['label']}: \
                            {element['value']}")
            for element in next_state_with_labels:
                logs.append(f"State {element['label']}: \
                            {element['value']}")

            for log in logs:
                print(log)

            print()

        if train:
            agent.train_short_memory(prev_state, action,
                                     reward, next_state, done)
            agent.remember(prev_state, action,
                           reward, next_state, done)

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
                # if train:
                #     agent.model.save(
                #         config['models_path'] + "best_model"
                #         + str(score) + ".pth"
                #     )

            if episodes is not None and games >= episodes:
                if train:
                    agent.model.save(
                        config['models_path'] +
                        "final_model"
                        + str(episodes) + "_episodes"
                        + "_best_score_" + str(best_score)
                        + "_mean_score_" + str(mean_scores[-1])
                        + ".pth"
                    )
                break


if __name__ == "__main__":
    main()
