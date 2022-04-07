import pygame
from defs import *

# Point inside rect check
def check_interacting(obj_dim, x, y):
    return (x >= obj_dim.x and x <= obj_dim.x + obj_dim.w and
            y >= obj_dim.y and y <= obj_dim.y + obj_dim.h) 

# TODO: Figure out if should delete number.jpg when clear canvas
def clear_canvas(surface, canvas_dim):
    pygame.draw.rect(surface, WHITE, canvas_dim)

def submit_canvas(surface, canvas_dim):
    sub = surface.subsurface(canvas_dim)
    pygame.image.save(sub, "number.jpg")

def main():
    # Initalize App Context
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill(WHITE)
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial.ttf', 32)
    running = True

    # Create Objects within app
    submit_button = Button(pygame.Rect(80, 360, 200, 80), (0, 255,  0), (150, 150, 150), font, 'Submit')
    clear_button = Button(pygame.Rect(360, 360, 200, 80), (255, 0,  0), (150, 150, 150), font, 'Clear')
    canvas_dim = pygame.Rect(0, 0, 320, 320)
    canvas_d_dim = pygame.Rect(0, 0, 315, 315)

    output = font.render('Output:', True, BLACK)

    while running:
        clock.tick(FPS)

        # Get inputs
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        left_down, _, right_down = pygame.mouse.get_pressed()
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Update application
        submit_button.hover = check_interacting(submit_button.dim, mouse_x, mouse_y)
        clear_button.hover = check_interacting(clear_button.dim, mouse_x, mouse_y)
        canvas_hover = check_interacting(canvas_d_dim, mouse_x, mouse_y)

        if left_down:
            if submit_button.hover:
                submit_canvas(screen, canvas_dim)
            elif clear_button.hover:
                clear_canvas(screen, canvas_dim)
            elif canvas_hover:
                pygame.draw.circle(screen, BLACK, (mouse_x, mouse_y), 5)

        elif right_down and canvas_hover:
                pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), 5)

        # Render applications
        pygame.draw.line(screen, BLACK, (0, 321), (640, 321))
        pygame.draw.line(screen, BLACK, (321, 0), (321, 321))
        submit_button.draw(screen)
        clear_button.draw(screen)

        screen.blit(output, (325, 5))

        pygame.display.flip()

if __name__ == '__main__':
    # Initialize application
    pygame.init()
    pygame.mixer.init()
    pygame.font.init()

    main()
    pygame.quit()

