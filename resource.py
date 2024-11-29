import pygame
import math

class Resource:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.max_amount = 100
        self.amount = self.max_amount
        self.is_exhausted = False
        self.collection_rate = 20  # Amount depleted per collection

    def collect(self):
        if self.amount >= self.collection_rate:
            self.amount -= self.collection_rate
            if self.amount <= 0:
                self.is_exhausted = True
            return True
        return False

    def draw(self, screen):
        # Draw the resource with size based on remaining amount
        depletion_ratio = self.amount / self.max_amount
        current_radius = max(8 * depletion_ratio, 2)  # Minimum size of 2

        if not self.is_exhausted:
            # Draw outer circle (resource boundary)
            pygame.draw.circle(screen, (139, 69, 19), (int(self.x), int(self.y)), 8)
            # Draw inner circle (filling based on amount)
            pygame.draw.circle(screen, (101, 67, 33), (int(self.x), int(self.y)), int(current_radius))
        else:
            # Draw depleted resource
            pygame.draw.circle(screen, (169, 169, 169), (int(self.x), int(self.y)), 4)
