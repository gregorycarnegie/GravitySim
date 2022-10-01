# This is a sample Python script.
import numpy as np
import time
import pygame

WHITE, YELLOW, BLUE, RED, DARK_GREY = (255, 255, 255), (255, 255, 0), (100, 149, 237), (188, 39, 50), (80, 78, 81)
Grav, AU, TIME_STEP = 6.67408e-11, 1.495978707e11, 10 * 3600 * 12
SCALE = 30 / AU  # 1 AU = 100px


def time_funk(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        print(f'{func.__module__}, {func.__name__}: {time.perf_counter() - start} seconds')
        return result

    return wrapper


@time_funk
def accel(pos, mass, G, softening):
    dx, dy, dz = [position.T - position for position in [pos[:, i:i+1] for i in range(3)]]
    inv_r3 = (dx ** 2 + dy ** 2 + dz ** 2 + softening ** 2) ** (-1.5)
    return np.hstack([G * (coordinate * inv_r3) @ mass for coordinate in [dx, dy, dz]])


def nbody(n, trail):
    pygame.init()
    WIDTH = HEIGHT = 2000
    colours = [WHITE, YELLOW, BLUE]
    WIN = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Planet Sim")
    posgen, velgen, masgen = np.random.default_rng(233423), np.random.default_rng(456452), np.random.default_rng(553452)

    positions = posgen.standard_normal(size=(n, 3), dtype=np.float64) * AU * 100
    vel = velgen.standard_normal(size=(n, 3), dtype=np.float64) * 1e8
    masses = np.absolute(masgen.standard_normal(size=(n, 1), dtype=np.float64)) * 2e32

    positions[0] = vel[0] = np.zeros(shape=(1, 3))
    masses[0] = 2e39

    dt = 10 * 3600 * 12
    dt /= 10000
    run = True
    clock = pygame.time.Clock()
    t = 0

    clock.tick(120)
    if trail:
        WIN.fill((0, 0, 0))

    while run:
        if not trail:
            WIN.fill((0, 0, 0))
        acc = accel(positions, masses, Grav, 0.001)
        i = 0
        for planet in positions:
            x, y = [int(w / AU) + (WIDTH * 0.5) for w in [planet[0], planet[1]]]
            if i == 0:
                pygame.draw.circle(WIN, RED, (x, y), 10)
            else:
                pygame.draw.circle(WIN, colours[i % len(colours)], (x, y), 1)
            i += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        positions += vel * dt + acc + dt * dt * 0.5
        vel += acc * dt

        t += dt

        pygame.display.update()


if __name__ == '__main__':
    trail_line = True
    nbody(32, trail_line)
