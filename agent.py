import pygame
import math
import random

class Agent:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.speed = 2
        self.radius = 5

    def move(self, dx, dy):
        self.x = max(0, min(800, self.x + dx * self.speed))
        self.y = max(0, min(600, self.y + dy * self.speed))

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def distance_to(self, other):
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

class Collector(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (255, 255, 0))
        self.carrying_resource = False
        self.target_resource = None
        self.state = 'idle'
        self.wait_time = 0
        self.collection_attempts = 0
        self.max_collection_attempts = 5  # Maximum attempts before finding new resource

    def update(self, resources, base, scouts):
        if self.wait_time > 0:
            self.wait_time -= 1
            return

        if self.state == 'idle':
            if not self.carrying_resource:
                discovered_resources = []
                for scout in scouts:
                    discovered_resources.extend(scout.found_resources)

                if discovered_resources:
                    # Filter out exhausted resources and those targeted by other collectors
                    available_resources = [r for r in discovered_resources
                                           if r in resources and not r.is_exhausted]
                    if available_resources:
                        self.target_resource = min(available_resources,
                                                   key=lambda r: self.distance_to(r))
                        self.state = 'collecting'
                        self.collection_attempts = 0
            else:
                self.state = 'returning'

        elif self.state == 'collecting':
            if self.target_resource and not self.target_resource.is_exhausted:
                # Move toward resource
                dx = self.target_resource.x - self.x
                dy = self.target_resource.y - self.y
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < 10:
                    if self.target_resource.collect():  # Try to collect
                        self.carrying_resource = True
                        self.color = (200, 200, 0)
                        self.state = 'returning'
                    else:
                        # Resource is exhausted
                        self.collection_attempts += 1
                        if self.collection_attempts >= self.max_collection_attempts:
                            self.target_resource = None
                            self.state = 'idle'
                else:
                    self.move(dx / dist, dy / dist)
            else:
                self.state = 'idle'

        elif self.state == 'returning':
            # Move toward base
            dx = base.x - self.x
            dy = base.y - self.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < 20:
                if self.carrying_resource:
                    self.carrying_resource = False
                    self.color = (255, 255, 0)
                    base.resources += 1
                    self.target_resource = None
                    self.state = 'idle'
                    self.wait_time = 20
            else:
                self.move(dx / dist, dy / dist)


class Scout(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (0, 255, 0))
        self.found_resources = []
        self.state = 'exploring'
        self.return_threshold = 400
        self.scan_radius = 50

        self.base_x = x
        self.base_y = y

        # Search pattern parameters
        self.search_mode = 'quadrant'
        self.spiral_angle = 0
        self.spiral_radius = 50
        self.spiral_step = 10
        self.min_distance_from_base = 50

        # Quadrant search parameters - store visited quadrants
        self.visited_quadrants = set()
        self.current_quadrant = random.randint(0, 3)
        self.quadrants = [
            (0, 0, 400, 300),    # top-left
            (400, 0, 800, 300),  # top-right
            (0, 300, 400, 600),  # bottom-left
            (400, 300, 800, 600) # bottom-right
        ]
        self.quadrant_points = []
        self.current_point_index = 0
        self.generate_new_quadrant_points()

    def generate_new_quadrant_points(self):
        quadrant = self.quadrants[self.current_quadrant]
        x1, y1, x2, y2 = quadrant
        points = []
        step = 100  # Increased step size for wider coverage

        # Calculate center of the current quadrant
        quadrant_center_x = (x1 + x2) / 2
        quadrant_center_y = (y1 + y2) / 2

        # Create a grid of points within the quadrant
        for x in range(int(x1 + step / 2), int(x2), step):
            for y in range(int(y1 + step / 2), int(y2), step):
                # Skip points too close to base
                dist_to_base = math.sqrt((x - self.base_x) ** 2 + (y - self.base_y) ** 2)
                if dist_to_base > self.min_distance_from_base:
                    points.append((x, y))

        # If no points were generated (all filtered out), add at least one point
        if not points:
            # Add a point at the quadrant center if it's far enough from base
            dist_to_base = math.sqrt((quadrant_center_x - self.base_x) ** 2 +
                                     (quadrant_center_y - self.base_y) ** 2)
            if dist_to_base > self.min_distance_from_base:
                points.append((quadrant_center_x, quadrant_center_y))
            else:
                # Add a point at the farthest corner from the base
                corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
                farthest_corner = max(corners,
                                      key=lambda p: math.sqrt((p[0] - self.base_x) ** 2 + (p[1] - self.base_y) ** 2))
                points.append(farthest_corner)

        # Randomize points to prevent predictable patterns
        random.shuffle(points)
        self.quadrant_points = points
        self.current_point_index = 0

    def switch_quadrant(self):
        self.current_quadrant = (self.current_quadrant+1) % 4
        self.generate_new_quadrant_points()

    def get_spiral_point(self):
        # Generate next point in spiral pattern
        self.spiral_angle += 0.2  # Slower spiral
        self.spiral_radius += self.spiral_step
        x = self.base_x + self.spiral_radius * math.cos(self.spiral_angle)
        y = self.base_y + self.spiral_radius * math.sin(self.spiral_angle)

        # Keep within bounds
        x = max(50, min(750, x))
        y = max(50, min(550, y))

        # If spiral gets too large, reset it
        if self.spiral_radius > 200:
            self.search_mode = 'quadrant'
            self.generate_new_quadrant_points()

        return x, y

    def get_next_search_point(self, base):
        if self.search_mode == 'spiral':
            return self.get_spiral_point()
        else:  # quadrant search
            if not self.quadrant_points or self.current_point_index >= len(self.quadrant_points):
                self.switch_quadrant()

            if self.quadrant_points:
                return self.quadrant_points[self.current_point_index]

        return base.x + self.min_distance_from_base, base.y  # fallback point

    def update(self, resources, base):
        # Update base coordinates if needed
        self.base_x = base.x
        self.base_y = base.y

        # Update found_resources to remove exhausted ones
        self.found_resources = [r for r in self.found_resources if not r.is_exhausted]

        # Scan for new resources
        for resource in resources:
            if (self.distance_to(resource) < self.scan_radius and
                    resource not in self.found_resources and
                    not resource.is_exhausted):
                self.found_resources.append(resource)
                # Switch to spiral search around the found resource
                self.search_mode = 'spiral'
                self.base_x = resource.x
                self.base_y = resource.y
                self.spiral_radius = self.scan_radius
                self.spiral_angle = 0

        if self.state == 'exploring':
            distance_to_base = self.distance_to(base)
            if distance_to_base > self.return_threshold:
                self.state = 'returning'
                return

            # Get next search point
            target_x, target_y = self.get_next_search_point(base)

            # Move towards target point
            dx = target_x - self.x
            dy = target_y - self.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 5:  # Only move if we're not very close to target
                self.move(dx / dist, dy / dist)
            else:
                # Move to next point
                if self.search_mode == 'quadrant':
                    self.current_point_index += 1
                    if self.current_point_index >= len(self.quadrant_points):
                        self.switch_quadrant()

                # Occasionally switch search modes
                if random.random() < 0.05:  # 5% chance to switch modes
                    if self.search_mode == 'spiral':
                        self.search_mode = 'quadrant'
                        self.generate_new_quadrant_points()
                    else:
                        self.search_mode = 'spiral'
                        self.spiral_radius = self.min_distance_from_base
                        self.spiral_angle = random.random() * 2 * math.pi

        else:  # returning state
            dx = base.x - self.x
            dy = base.y - self.y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0:
                self.move(dx / dist, dy / dist)

            if dist < 30:
                self.state = 'exploring'
                self.search_mode = 'quadrant'
                self.generate_new_quadrant_points()

    def draw(self, screen):
        super().draw(screen)
        # Draw scan radius
        pygame.draw.circle(screen, (0, 255, 0),
                           (int(self.x), int(self.y)),
                           self.scan_radius, 1)

        # Draw target point
        if self.state == 'exploring':
            if self.search_mode == 'quadrant' and self.quadrant_points and self.current_point_index < len(
                    self.quadrant_points):
                target = self.quadrant_points[self.current_point_index]
                pygame.draw.circle(screen, (0, 100, 0),
                                   (int(target[0]), int(target[1])), 3)


class Guardian(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (0, 0, 255))
        self.detection_radius = 100

    def detect_intruders(self, intruders):
        detected = []
        for intruder in intruders:
            if self.distance_to(intruder) < self.detection_radius:
                detected.append(intruder)
        return detected


class Intruder(Agent):
    def __init__(self, x, y):
        super().__init__(x, y, (255, 0, 0))
