import math
import time
from collections import deque

import cv2
import keyboard
import mss
import numpy as np

# --- Configuration ---
SCREEN_W, SCREEN_H = 2560, 1440
CROP_W, CROP_H = 200, 200
REGION = {
    "left": (SCREEN_W - CROP_W) // 2,
    "top": (SCREEN_H - CROP_H) // 2,
    "width": CROP_W,
    "height": CROP_H,
}
FRAME_RATE = 120
REPAIR_SPEED = 330
WIGGLE_SPEED = 230
PRESS_DELAY = 0.0032
DELAY_PIXEL = 8
RED_HSV_LOWER = np.array([0, 100, 100])
RED_HSV_UPPER = np.array([10, 255, 255])
WHITE_BGR_LOWER = np.array([240, 240, 240])
WHITE_BGR_UPPER = np.array([255, 255, 255])

# Precompute circular mask to limit detection within circle
_circle_mask = np.zeros((CROP_H, CROP_W), dtype=np.uint8)
cv2.circle(_circle_mask, (CROP_W // 2, CROP_H // 2), CROP_W // 2, 255, -1)


def get_screenshot():
    with mss.mss() as sct:
        sct_img = sct.grab(REGION)
        img = np.array(sct_img)[:, :, :3]  # RGBA->BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img


recent_red = deque(maxlen=3)


def find_thickest(mask):
    # morphology: find largest square-inscribed point
    max_r = None
    max_d = 0
    # try increasing kernel sizes
    for d in range(1, 21):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * d + 1, 2 * d + 1))
        eroded = cv2.erode(mask, kernel)
        if eroded.any():
            # get coords of remaining pixels
            ys, xs = np.where(eroded > 0)
            # pick center pixel arbitrarily
            max_r = (ys.mean(), xs.mean())
            max_d = d
        else:
            break
    if max_r is None:
        return None, None, 0
    return int(max_r[0]), int(max_r[1]), max_d


def find_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, RED_HSV_LOWER, RED_HSV_UPPER)
    # limit to circular region
    mask = cv2.bitwise_and(mask1, mask1, mask=_circle_mask)
    if not mask.any():
        return None
    yi, xi, d = find_thickest(mask)
    if d < 1:
        return None
    return (yi, xi, d)


def find_square(img):
    # threshold white in BGR
    mask1 = cv2.inRange(img, WHITE_BGR_LOWER, WHITE_BGR_UPPER)
    # mask out center zone to ignore UI
    cx, cy = CROP_W // 2, CROP_H // 2
    cv2.circle(mask1, (cx, cy), 40, 0, -1)
    if not mask1.any():
        return None
    yi, xi, d = find_thickest(mask1)
    if d < 1:
        return None
    # compute endpoints along direction
    theta = math.atan2(yi - cy, xi - cx)
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    pre = (int(yi - sin_t * d), int(xi - cos_t * d))
    post = (int(yi + sin_t * d), int(xi + cos_t * d))
    mid = ((pre[0] + post[0]) // 2, (pre[1] + post[1]) // 2)
    # verify mid is white
    if mask1[mid] == 0:
        return None
    return mid, pre, post


def press_space_at(t_click):
    delay = t_click - time.time()
    if delay > -0.05:
        if delay > 0:
            time.sleep(delay)
        keyboard.press_and_release("c")
        return True
    return False


def timer_loop():
    last_t = time.time()
    while True:
        img1 = get_screenshot()
        t1 = time.time()
        red1 = find_red(img1)
        if not red1:
            continue
        recent_red.append(red1)
        if sum(1 for e in recent_red if e) < 2:
            continue
        yi1, xi1, d1 = red1
        img2 = get_screenshot()
        red2 = find_red(img2)
        if not red2:
            continue
        yi2, xi2, d2 = red2
        if (yi2 - xi2) == (yi1 - xi1):
            continue
        # direction
        delta_deg = math.degrees(
            math.atan2(yi2 - CROP_H / 2, xi2 - CROP_W / 2)
        ) - math.degrees(math.atan2(yi1 - CROP_H / 2, xi1 - CROP_W / 2))
        direction = 1 if delta_deg % 360 < 180 else -1
        # detect white
        square = find_square(img1)
        if not square:
            continue
        white, pre, post = square
        target_deg = math.degrees(
            math.atan2(white[0] - CROP_H / 2, white[1] - CROP_W / 2)
        )
        speed = REPAIR_SPEED * direction
        delta = (
            target_deg - math.degrees(math.atan2(yi1 - CROP_H / 2, xi1 - CROP_W / 2))
        ) % (360 * direction)
        t_click = t1 + delta / abs(speed) - PRESS_DELAY
        if press_space_at(t_click):
            print(f"Fired at {t_click - t1:.3f}s")


if __name__ == "__main__":
    timer_loop()
