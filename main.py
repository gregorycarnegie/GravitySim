import numpy as np
import pygame

WHITE, YELLOW, BLUE, RED, DARK_GREY = (255, 255, 255), (255, 255, 0), (100, 149, 237), (188, 39, 50), (80, 78, 81)
GRAV, AU, TIME_STEP = 6.67408e-11, 1.495978707e11, 10 * 3_600 * 12 / 10_000
SCALE = 30 / AU  # 1 AU = 100px

def accel(pos: np.random._generator.Generator,
          mass: np.ndarray,
          g: float=GRAV,
          softening: float=0.001) -> np.ndarray:
    dx, dy, dz = [position.T - position for position in [pos[:, i:i+1] for i in range(3)]]
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2) ** (-1.5)
    return np.hstack([g * (coordinate * inv_r3) @ mass for coordinate in [dx, dy, dz]])

def nbody(n: int=128, trail: bool=False) -> None:
    pygame.init()
    WIDTH = HEIGHT = 2000
    colours = [WHITE, YELLOW, BLUE]
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Planet Sim")
    pos_gen, vel_gen, mas_gen = (np.random.default_rng(x) for x in (233_423, 456_452, 553_452))

    positions = pos_gen.standard_normal(size=(n, 3), dtype=np.float64) * AU * 100
    vel = vel_gen.standard_normal(size=(n, 3), dtype=np.float64) * 5e7
    masses = np.absolute(mas_gen.standard_normal(size=(n, 1), dtype=np.float64)) * 2e32

    positions[0] = vel[0] = np.zeros((1, 3))
    masses[0] = 2e39 # Mass od main body e.g. the Sun, Black Hole etc.

    run = True
    clock = pygame.time.Clock()
    t = 0
    clock.tick(120)

    if trail:
        WIN.fill((0, 0, 0))

    while run:
        if not trail:
            WIN.fill((0, 0, 0))
        acc = accel(positions, masses)
        for i, planet in enumerate(positions):
            x, y = planet[:2] / AU + WIDTH * 0.5
            if i == 0:
                pygame.draw.circle(WIN, RED, (x, y), 10)
            else:
                pygame.draw.circle(WIN, colours[i % len(colours)], (x, y), 1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        positions += vel * TIME_STEP + acc + TIME_STEP * TIME_STEP * 0.5
        vel += acc * TIME_STEP
        t += TIME_STEP
        pygame.display.update()

if __name__ == '__main__':
    nbody()
