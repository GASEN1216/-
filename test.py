import pyautogui
import cv2
import numpy as np
import time
from collections import deque

# 设置捕获区域为屏幕中心附近的一块区域
CENTER_X, CENTER_Y = 960, 540
REGION_WIDTH, REGION_HEIGHT = 100, 100
GAME_REGION = (CENTER_X - REGION_WIDTH // 2, CENTER_Y - REGION_HEIGHT // 2, REGION_WIDTH, REGION_HEIGHT)

# 鱼标和亮绿色条的 HSV 范围
fish_lower = np.array([80, 150, 150])  # 鱼标 HSV 下限
fish_upper = np.array([105, 255, 255])  # 鱼标 HSV 上限

bright_green_lower = np.array([10, 50, 0])  # 亮绿色 HSV 下限
bright_green_upper = np.array([85, 255, 255])  # 亮绿色 HSV 上限

# 钓鱼条 UI 模板路径
FISHING_UI_TEMPLATE = "fishing_ui.png"  # 请确保路径正确

# 记录历史位置的队列长度
HISTORY_LENGTH = 5


def capture_screen(region=None):
    """捕获屏幕指定区域的截图"""
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def detect_fishing_ui(frame):
    """检测钓鱼条 UI 的位置"""
    template = cv2.imread(FISHING_UI_TEMPLATE, cv2.IMREAD_GRAYSCALE)

    if template is None:
        print(f"[错误] 无法加载模板图像 '{FISHING_UI_TEMPLATE}'。请检查路径是否正确。")
        return None

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 执行模板匹配
    res = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        x, y = pt
        w, h = template.shape[::-1]
        # print(f"[成功] 检测到钓鱼条 UI，位置: x={x}, y={y}, 宽度={w}, 高度={h}")
        return x, y, w, h  # 返回 UI 的位置和大小

    # print("[警告] 未检测到钓鱼条 UI，请确保游戏界面已显示。")
    return None


def find_fish_excluding_green(frame):
    """检测鱼标位置并排除亮绿色条"""
    # 转换图像到 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建鱼标的掩膜
    fish_mask = cv2.inRange(hsv, fish_lower, fish_upper)

    # 创建亮绿色的掩膜
    bright_green_mask = cv2.inRange(hsv, bright_green_lower, bright_green_upper)

    # 排除亮绿色区域
    fish_mask = cv2.bitwise_and(fish_mask, cv2.bitwise_not(bright_green_mask))

    # 查找轮廓
    contours, _ = cv2.findContours(fish_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓：根据面积和长宽比筛选轮廓
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 0:  # 过滤掉小的噪点轮廓
            x, y, w, h = cv2.boundingRect(contour)
            filtered_contours.append((contour, (x, y, w, h)))

    # 如果有符合条件的轮廓，返回最大的轮廓的位置和掩膜
    if filtered_contours:
        largest_contour, (x, y, w, h) = max(filtered_contours, key=lambda c: cv2.contourArea(c[0]))
        fish_position = (x + w // 2, y + h // 2)  # 计算轮廓的中心点

        # 以中心点为基准绘制矩形并填充红色
        rect_width = 25  # 自定义矩形的宽度
        rect_height = 40  # 自定义矩形的高度

        # 计算矩形的左上角和右下角坐标
        top_left = (fish_position[0] - rect_width // 2, fish_position[1] - rect_height // 2)
        bottom_right = (fish_position[0] + rect_width // 2, fish_position[1] + rect_height // 2)

        # 绘制并填充矩形
        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), -1)  # -1 表示填充矩形

        # 在正方形中心绘制一个小圆点，标记中心位置
        cv2.circle(frame, fish_position, 5, (0, 0, 255), -1)

        return fish_position, fish_mask

    return None, fish_mask


def find__green(frame):
    """检测钓鱼条位置"""
    # 转换图像到 HSV 颜色空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 创建鱼标的掩膜
    fish_mask = cv2.inRange(hsv, fish_lower, fish_upper)

    # 创建亮绿色的掩膜
    bright_green_mask = cv2.inRange(hsv, bright_green_lower, bright_green_upper)

    # 排除亮绿色区域
    bright_green_mask = cv2.bitwise_and(bright_green_mask, cv2.bitwise_not(fish_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))  # 针对垂直条状特征
    processed_mask = cv2.morphologyEx(bright_green_mask, cv2.MORPH_CLOSE, kernel)

    # 查找形态学处理后的轮廓
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤轮廓：根据面积筛选轮廓
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # 如果有符合条件的轮廓，返回最大的轮廓的位置和掩膜
    if filtered_contours:
        # 合并所有轮廓
        combined_contours = np.vstack(filtered_contours)

        x, y, w, h = cv2.boundingRect(combined_contours)
        bright_green_position = (x + w // 2, y + h // 2)

        cv2.drawContours(frame, [combined_contours], -1, (0, 255, 0), 2)
        cv2.circle(frame, bright_green_position, 5, (0, 0, 255), -1)

        return bright_green_position, bright_green_mask

    return None, bright_green_mask


def calculate_speed(history):
    """根据历史位置计算速度"""
    if len(history) >= 2:
        return history[-1] - history[-2]
    return 0

def control_fishing_bar(fish_y, fish_speed, green_y, green_speed):
    """根据鱼标和绿色条的位置及速度控制鼠标"""
    diff = fish_y - green_y

    # 如果鱼标在绿色条上方且绿色条速度较大，快速点击来控制上升幅度
    if diff < 0:
        if green_speed > fish_speed:
            pyautogui.click()
        else:
            pyautogui.mouseDown()
    # 如果鱼标在绿色条下方且绿色条速度较大，松开鼠标来控制下落
    elif diff > 0:
        if green_speed > fish_speed:
            pyautogui.mouseUp()
        else:
            pyautogui.click()
    else:
        pyautogui.click()  # 保持微调


def main():
    print("准备就绪，开始检测鱼标...")

    fish_history = deque(maxlen=HISTORY_LENGTH)
    green_history = deque(maxlen=HISTORY_LENGTH)

    try:
        while True:
            frame = capture_screen(GAME_REGION)

            # 检测钓鱼条 UI
            ui_region = detect_fishing_ui(frame)

            if ui_region:
                # 重新截取钓鱼条 UI 区域的图像
                x, y, w, h = ui_region
                new_region = (
                    int(GAME_REGION[0] + x + 33),
                    int(GAME_REGION[1] + y + 115),
                    int(w - 10),
                    int(h + 15)
                )
                new_frame = capture_screen(new_region)

                # 检测鱼标和绿色条的位置
                fish_position, mask = find_fish_excluding_green(new_frame)
                green_position, green_mask = find__green(new_frame)

                if fish_position and green_position:
                    fish_y = fish_position[1]
                    green_y = green_position[1]

                    # 记录历史位置
                    fish_history.append(fish_y)
                    green_history.append(green_y)

                    # 计算速度
                    fish_speed = calculate_speed(fish_history)
                    green_speed = calculate_speed(green_history)

                    # 控制绿色条
                    control_fishing_bar(fish_y, fish_speed, green_y, green_speed)

                # 显示检测结果
                cv2.imshow("Fish Mask (Excluding Green)", mask)
                cv2.imshow("Green Mask", green_mask)
                cv2.imshow("Screen Capture", new_frame)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("检测停止。")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
