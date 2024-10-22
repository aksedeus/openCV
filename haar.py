from __future__ import print_function
import cv2 as cv
import numpy as np
import pyautogui
import time

def detectAndDisplay(frame):
    # Преобразуем кадр в оттенки серого
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Выравниваем гистограмму для улучшения контрастности
    frame_gray = cv.equalizeHist(frame_gray)

    # -- Обнаружение лиц
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        center = (x + w // 2, y + h // 2)
        # Рисуем эллипс вокруг обнаруженного лица
        frame = cv.ellipse(frame, center, (w // 2, h // 2), 0, 0, 360, (255, 0, 255), 4)

        faceROI = frame_gray[y:y + h, x:x + w]
        # -- В каждом лице обнаруживаем глаза
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            eye_center = (x + x2 + w2 // 2, y + y2 + h2 // 2)
            radius = int(round((w2 + h2) * 0.25))
            # Рисуем круг вокруг обнаруженного глаза
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0), 4)

    # Отображаем результат
    cv.imshow('Capture - Face detection', frame)

# Загружаем каскады
face_cascade_name = 'data/haarcascades/haarcascade_frontalface_alt.xml'
eyes_cascade_name = 'data/haarcascades/haarcascade_eye_tree_eyeglasses.xml'

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

# Проверяем загрузку каскадов
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    print('--(!) Ошибка загрузки каскада лиц')
    exit(0)
if not eyes_cascade.load(cv.samples.findFile(eyes_cascade_name)):
    print('--(!) Ошибка загрузки каскада глаз')
    exit(0)

while True:
    # Захватываем экран
    screenshot = pyautogui.screenshot()

    # Конвертируем скриншот в массив numpy
    frame = np.array(screenshot)

    # Конвертируем из RGB в BGR (OpenCV использует BGR)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # Обрабатываем кадр
    detectAndDisplay(frame)

    # Ожидаем 10 мс и выходим, если нажата клавиша 'ESC'
    if cv.waitKey(10) == 27:
        break

# Закрываем все окна
cv.destroyAllWindows()
