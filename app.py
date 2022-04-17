import pygame
import cv2
from keras.models import load_model
import numpy as np
import os
import math
from PIL import Image
from defs import *

model = load_model('model/dig_rec_classifer.h5')

def predict_digit(transformed_img):
    test_image = transformed_img.reshape(-1,28,28,1)
    return np.argmax(model.predict(test_image))

def image_refiner(input_img):
    org_size = 22
    img_size = 28
    rows, cols = input_img.shape
    
    if rows > cols:
        factor = org_size/rows
        rows = org_size
        cols = int(round(cols*factor))        
    else:
        factor = org_size/cols
        cols = org_size
        rows = int(round(rows*factor))
    input_img = cv2.resize(input_img, (cols, rows))
    
    col_padding = (int(math.ceil((img_size-cols)/2.0)), int(math.floor((img_size-cols)/2.0)))
    row_padding = (int(math.ceil((img_size-rows)/2.0)), int(math.floor((img_size-rows)/2.0)))
    
    input_img = np.lib.pad(input_img, (row_padding, col_padding), 'constant')
    return input_img

def process_img(path_to_img):
    img = cv2.imread(path_to_img, 2)
    img_org =  cv2.imread(path_to_img)

    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for j, count in enumerate(contours):
        epsilon = 0.01 * cv2.arcLength(count, True)
        approx = cv2.approxPolyDP(count, epsilon, True)
        
        hull = cv2.convexHull(count)
        k = cv2.isContourConvex(count)
        x, y, w, h = cv2.boundingRect(count)
        
        if (hierarchy[0][j][3]!=-1 and w > 10 and h > 10):
            processed = img[y:y+h, x:x+w]
            processed = cv2.bitwise_not(processed)
            processed = image_refiner(processed)
            return processed

    return None

# Point inside rect check
def check_interacting(obj_dim, x, y):
    return (x >= obj_dim.x and x <= obj_dim.x + obj_dim.w and
            y >= obj_dim.y and y <= obj_dim.y + obj_dim.h) 

def clear_canvas(surface, canvas_dim):
    pygame.draw.rect(surface, WHITE, canvas_dim)

def submit_canvas(surface, canvas_dim):
    sub = surface.subsurface(canvas_dim)
    pygame.image.save(sub, "number.jpg")

    path_to_img = os.path.join(os.getcwd(), "number.jpg")
    transformed_img = process_img("number.jpg")
    digit = predict_digit(transformed_img)

    return digit

def main():
    # Initalize App Context
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    screen.fill(WHITE)
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial.ttf', 32)
    font_big = pygame.font.SysFont('Arial.ttf', 128)
    running = True
    prev_down = False

    # Create Objects within app
    submit_button = Button(pygame.Rect(80, 360, 200, 80), (0, 255,  0), (150, 150, 150), font, 'Submit')
    clear_button = Button(pygame.Rect(360, 360, 200, 80), (255, 0,  0), (150, 150, 150), font, 'Clear')
    canvas_dim = pygame.Rect(0, 0, 320, 320)
    canvas_d_dim = pygame.Rect(0, 0, 315, 315)
    output_dim = pygame.Rect(321, 1, 319, 319)

    output = font.render('Output:', True, BLACK)
    digit_label = font.render('', True, BLACK)

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
            if submit_button.hover and not prev_down:
                clear_canvas(screen, output_dim)
                digit = submit_canvas(screen, canvas_dim)
                digit_label = font_big.render(str(digit), True, BLACK)
            elif clear_button.hover and not prev_down:
                clear_canvas(screen, canvas_dim)
            elif canvas_hover:
                pygame.draw.circle(screen, BLACK, (mouse_x, mouse_y), 10)

        elif right_down and canvas_hover:
                pygame.draw.circle(screen, WHITE, (mouse_x, mouse_y), 10)
        
        prev_down = left_down

        # Render applications
        pygame.draw.line(screen, BLACK, (0, 321), (640, 321))
        pygame.draw.line(screen, BLACK, (321, 0), (321, 321))
        submit_button.draw(screen)
        clear_button.draw(screen)

        screen.blit(output, (325, 5))
        screen.blit(digit_label, (450, 100))

        pygame.display.flip()

if __name__ == '__main__':
    # Initialize application
    pygame.init()
    pygame.mixer.init()
    pygame.font.init()

    main()
    pygame.quit()

