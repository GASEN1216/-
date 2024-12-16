import cv2
import keyboard
import numpy as np
import pyautogui
import time

from test import capture_screen, GAME_REGION, detect_fishing_ui

# 黄色的 HSV 范围
YELLOW_LOWER = np.array([20, 100, 200])  # 黄色的 HSV 下界
YELLOW_UPPER = np.array([30, 255, 255])  # 黄色的 HSV 上界

def find_yellow(frame):
    """检测黄色区域并返回掩膜和是否检测到黄色"""
    # 转换图像到 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建黄色掩膜
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)

    # 查找黄色区域的轮廓
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 如果检测到黄色区域，返回 True 和掩膜
    if contours:
        return True, yellow_mask
    return False, yellow_mask

def perform_mouse_action():
    """按下鼠标0.9秒后松开，然后开始检测黄色"""
    time.sleep(2)  # 防止连续触发
    pyautogui.mouseDown()
    pyautogui.mouseUp()
    print("按下鼠标...")
    pyautogui.mouseDown()
    time.sleep(0.92)
    pyautogui.mouseUp()
    print("松开鼠标")

    while True:
        # 捕获屏幕指定区域
        frame = capture_screen(GAME_REGION)

        # 检测黄色区域
        detected, mask = find_yellow(frame)

        # 显示原始截取的画面和黄色掩膜
        cv2.imshow("Screen Capture", frame)
        cv2.imshow("Yellow Mask", mask)

        # 如果检测到黄色，停止检测并退出循环
        if detected:
            print("检测到黄色，停止检测。")
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            break

        # 按 'q' 键强制退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("手动退出检测。")
            break

    cv2.destroyAllWindows()

def main():
    print("程序启动，按下 'o' 键来触发操作。按 'p' 键退出程序。")

    while True:
        if keyboard.is_pressed('o'):
            while True:
                perform_mouse_action()

                CENTER_X, CENTER_Y = 960, 540
                REGION_WIDTH, REGION_HEIGHT = 400, 1080
                GAME_REGION2 = (CENTER_X - REGION_WIDTH // 2, CENTER_Y - REGION_HEIGHT // 2, REGION_WIDTH, REGION_HEIGHT)
                frame = capture_screen(GAME_REGION2)
                ui_region = detect_fishing_ui(frame)
                if ui_region:
                    break

        if keyboard.is_pressed('q'):
            print("退出程序。")
            break

if __name__ == "__main__":
    main()
