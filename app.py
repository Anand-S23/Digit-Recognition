import pygame

# Settings
WIDTH = 640
HEIGHT = 480
FPS = 60
TITLE = "Digit Recognition"

# Definitions
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

def main(screen, clock):
    running = True

    while running:
        clock.tick(FPS)

        # Get inputs
        for event in pygame.event.get() :
            if event.type == QUIT :
                running = False

        # Update application

        # Render applications
        screen.fill(WHITE)
        pygame.display.flip()

if __name__ == '__main__':
    # Initialize application
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    main(screen, clock)

    pygame.quit()
