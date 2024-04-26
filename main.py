# este funcionó finalmente
import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
from pynput.keyboard import Key, Controller

import pygame
import random

# Configuración de pantalla
WIDTH, HEIGHT = 600, 400
GRID_SIZE = 20
GRID_WIDTH = WIDTH // GRID_SIZE
GRID_HEIGHT = HEIGHT // GRID_SIZE
FPS = 5

# Colores
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Direcciones
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

#defi
# definimos un tipo de objeto serpiente que tendrá varios atributos y métodos
class Snake:
    def __init__(self):
        self.body = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.grow = False

    def move(self):
        cur = self.body[0]
        x, y = self.direction
        new = (((cur[0] + (x * GRID_SIZE)) % WIDTH), (cur[1] + (y * GRID_SIZE)) % HEIGHT)
        if len(self.body) > 2 and new in self.body[2:]:
            self.reset()
        else:
            self.body = [new] + self.body[:-1]
            if self.grow:
                self.body.append(self.body[-1])
                self.grow = False

    def reset(self):
        self.body = [((WIDTH // 2), (HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def grow_snake(self):
        self.grow = True

    def draw(self, surface):
        for segment in self.body:
            pygame.draw.rect(surface, GREEN, (segment[0], segment[1], GRID_SIZE, GRID_SIZE))


class Food:
    def __init__(self):
        self.x = random.randint(0, GRID_WIDTH - 1) * GRID_SIZE
        self.y = random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE

    def draw(self, surface):
        pygame.draw.rect(surface, RED, (self.x, self.y, GRID_SIZE, GRID_SIZE))


# creamos un objeto que encontraran los puntos de las manos
mp_hands = mp.solutions.hands
# creamos un objeto que diburá las manos encontradas
mp_drawing = mp.solutions.drawing_utils

with mp_hands.Hands(static_image_mode=True,
                    max_num_hands=2,
                    min_detection_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    keyboard = Controller()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    snake = Snake()
    food = Food()

    Running = True

    while Running:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (720, 480), interpolation=cv2.INTER_NEAREST)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (5, 5), (715, 155), (255, 255, 255), 3)  # arriba
        cv2.rectangle(image, (5, 155), (235, 315), (255, 255, 255), 3)  # izquierda
        cv2.rectangle(image, (485, 155), (715, 315), (255, 255, 255), 3)  # derecha
        cv2.rectangle(image, (3, 325), (715, 475), (255, 255, 255), 3)  # abajo

        if results.multi_hand_landmarks:
            txt = ''
            key_to_press = ''
            for hand_landmark in results.multi_hand_landmarks:
                x1 = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].x * 720)
                y1 = int(hand_landmark.landmark[mp_hands.HandLandmark.WRIST].y * 480)
                cv2.circle(image, (x1, y1), 15, (255, 255, 255), -1)

                key_to_press = ''
                # ==================================================================
                # Comparación de dedos
                # ==================================================================

                # Verificar si (x1, y1) se encuentra en alguno de los rangos
                arriba = 5 < x1 < 715 and 5 < y1 < 155
                izquierda = 5 < x1 < 235 and 155 < y1 < 315
                derecha = 485 < x1 < 715 and 155 < y1 < 315
                abajo = 3 < x1 < 715 and 325 < y1 < 475

                # ==================================================================
                # Condiciones
                # ==================================================================

                if arriba:
                    # print('press arriba')
                    snake.direction = UP

                if izquierda:
                    # print('press izquierda')
                    snake.direction = LEFT

                if derecha:
                    # print('press derecha')
                    snake.direction = RIGHT

                if abajo:
                    # print('press abajo')
                    snake.direction = DOWN

                # ==================================================================
                # Acciones
                # ==================================================================

                # if key_to_press:
                #    keyboard.press(key_to_press)
                #    keyboard.release(key_to_press)

                snake.move()

        if snake.body[0] == (food.x, food.y):
            food = Food()
            snake.grow_snake()

        screen.fill((0, 0, 0))
        snake.draw(screen)
        food.draw(screen)
        pygame.display.update()
        clock.tick(FPS)

        cv2.imshow("Deteccion de la mano", image)
        key_to_press = ''
        if cv2.waitKey(1) & 0xFF == ord('q'):
            pygame.quit()
            Running = False
            break

    cap.release()
    cv2.destroyAllWindows()