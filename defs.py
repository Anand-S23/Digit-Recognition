import pygame

# Settings
WIDTH = 640
HEIGHT = 480
FPS = 120
TITLE = "Digit Recognition"

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
RED = (255, 0, 0)
GREEN = (0, 255, 0);

class Button():
    def __init__(self, dim, color, hover_color, font, text):
        self.dim = dim
        self.color = color
        self.hover_color = hover_color
        self.hover = False
        self.text = font.render(text, True, BLACK)

    def draw(self, surface):
        if self.hover:
            pygame.draw.rect(surface, self.hover_color, self.dim)
        else:
            pygame.draw.rect(surface, self.color, self.dim)
        
        x_pos = self.dim.center[0] - (self.text.get_width() / 2)
        y_pos = self.dim.center[1] - (self.text.get_height() / 2)
        surface.blit(self.text, (x_pos, y_pos))

