import pygame
from snake_env import SnakeGame

# Mapping von pygame-Tasten zu Actions
KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3
}

if __name__ == "__main__":
    game = SnakeGame(grid_size=25)
    state = game.reset()

    running = True
    action = 1  # Startbewegung nach rechts

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game.close()
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]

        state, reward, done, _ = game.step(action)

        if done:
            print("Game Over – Score:", game.score)
            pygame.time.wait(2000)  # Pause vor dem Schließen
            running = False
            game.close()
