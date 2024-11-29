import pygame
class Base:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.resources = 0

    def draw(self, screen):
        pygame.draw.rect(screen, (128, 128, 128), (self.x - 20, self.y - 20, 40, 40))

