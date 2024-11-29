import pygame
from agent import Collector, Scout, Intruder, Guardian
from resource import Resource
from replay_memory import ReplayMemory
from dqn import DQN
from base import Base
import math
import random

class HiveSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Hive Simulation")
        self.clock = pygame.time.Clock()

        # Initialize game objects
        self.base = Base(400, 300)
        self.scouts = [Scout(400, 300) for _ in range(3)]
        self.collectors = [Collector(400, 300) for _ in range(5)]
        self.guardians = [Guardian(400 + 50 * math.cos(i * 2 * math.pi / 4),
                                   300 + 50 * math.sin(i * 2 * math.pi / 4))
                          for i in range(4)]
        self.intruders = []
        self.resources = [Resource(random.randint(50, 750), random.randint(50, 550))
                          for _ in range(5)]

        self.running = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def spawn_intruder(self):
        if random.random() < 0.01:
            side = random.choice(['top', 'bottom', 'left', 'right'])
            if side == 'top':
                self.intruders.append(Intruder(random.randint(0, 800), 0))
            elif side == 'bottom':
                self.intruders.append(Intruder(random.randint(0, 800), 600))
            elif side == 'left':
                self.intruders.append(Intruder(0, random.randint(0, 600)))
            else:
                self.intruders.append(Intruder(800, random.randint(0, 600)))

    def update(self):
        # Update scouts
        for scout in self.scouts:
            scout.update(self.resources, self.base)

        # Update collectors
        for collector in self.collectors:
            collector.update(self.resources, self.base, self.scouts)

        # Update guardians and intruders
        for guardian in self.guardians:
            detected = guardian.detect_intruders(self.intruders)
            for intruder in detected:
                dx = intruder.x - guardian.x
                dy = intruder.y - guardian.y
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    guardian.move(dx / dist, dy / dist)
                if dist < 10 and intruder in self.intruders:
                    self.intruders.remove(intruder)

        # Update intruders
        self.spawn_intruder()
        for intruder in self.intruders:
            dx = self.base.x - intruder.x
            dy = self.base.y - intruder.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                intruder.move(dx / dist, dy / dist)

    def draw(self):
        self.screen.fill((255, 255, 255))

        # Draw all game objects
        self.base.draw(self.screen)
        for resource in self.resources:
            resource.draw(self.screen)
        for scout in self.scouts:
            scout.draw(self.screen)
        for collector in self.collectors:
            collector.draw(self.screen)
        for guardian in self.guardians:
            guardian.draw(self.screen)
        for intruder in self.intruders:
            intruder.draw(self.screen)

        # Draw resource count
        font = pygame.font.Font(None, 36)
        text = font.render(f'Resources: {self.base.resources}', True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    simulation = HiveSimulation()
    simulation.run()
