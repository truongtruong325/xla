import cv2
import numpy as np
import utlis
import serial
import time

# Thiết lập cổng serial
arduino_port = "COM5"  # Thay đổi cổng COM này tương ứng với Arduino của bạn
baud_rate = 9600       # Tốc độ baud phải khớp với cài đặt trên Arduino

# Mở kết nối serial
# ser = serial.Serial(arduino_port, baud_rate, timeout=1)

time.sleep(2)  # Chờ 2 giây để đảm bảo kết nối serial đã sẵn sàng

print("Sending data to Arduino...")

#Step 1
def preprocess(img):
    """Tiền xử lý ảnh để lấy ảnh từ góc nhìn từ trên cao của làn đường"""

    #### STEP 1
    imgThres = utlis.thresholding(img)
    cv2.imshow('threshol', imgThres)
    # img = grayscale(img)
    img_gauss = cv2.GaussianBlur(imgThres, (11, 11), 0)
    thresh_low = 150
    thresh_high = 200
    img_canny = cv2.Canny(img_gauss, thresh_low, thresh_high)
    return img_canny
    cv2.imshow("Canny", img_canny)
#step 2
def birdview_transform(img):
    """Lấy ảnh từ góc nhìn từ trên cao"""
    IMAGE_H = 480
    IMAGE_W = 500
    src = np.float32([[0, IMAGE_H], [640, IMAGE_H], [0, IMAGE_H * 0.4], [IMAGE_W, IMAGE_H * 0.4]])
    dst = np.float32([[240, IMAGE_H], [640 - 240, IMAGE_H], [-160, 0], [IMAGE_W + 160, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(img, M, (IMAGE_W, IMAGE_H))
    return warped_img

#step 3

def find_lane_lines(image, draw=False):
    """Tìm làn đường từ ảnh màu"""
    image = preprocess(image)
    im_height, im_width = image.shape[:2]
    if draw:
        viz_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#  xác định trung tâm làn đường
    interested_line_y = int(im_height * 0.8)
    if draw:
        cv2.line(viz_img, (0, interested_line_y),
                 (im_width, interested_line_y), (0, 0, 255), 2)
    interested_line = image[interested_line_y, :]

    # Xác định điểm trái và phải
    left_point = -1
    right_point = -1
    lane_width = 500

    center = im_width // 2
    for x in range(center, 0, -1):
        if interested_line[x] > 0:
            left_point = x
            break
    for x in range(center + 1, im_width):
        if interested_line[x] > 0:
            right_point = x
            break

    # Dự đoán điểm bị che khuất
    if left_point != -1 and right_point == -1:
        right_point = left_point + lane_width
      #   right_point = center
    if left_point == -1 and right_point != -1:
        left_point = right_point - lane_width
        # left_point = center
#vẽ điển lên kết quả
    if draw:
        if center != -1:
            viz_img = cv2.circle(
                viz_img, (center, interested_line_y), 7, (0, 0, 255), -1)
        if left_point != -1:
            viz_img = cv2.circle(
                viz_img, (left_point, interested_line_y), 7, (255, 0, 0), -1)
        if right_point != -1:
            viz_img = cv2.circle(
                viz_img, (right_point, interested_line_y), 7, (0, 255, 0), -1)

    if draw:
        return left_point, right_point, center, viz_img
    else:
        return left_point, right_point, center

#step 4
def calculate_control_signal(left_point, right_point, im_center):
    """Tính toán tín hiệu điều khiển"""

    if left_point == -1 or right_point == -1:
        #ser.write(('D' + "\n").encode())
        print("steering controll: dừng")
        return left_motor_speed, right_motor_speed

    # Tính toán sự khác biệt giữa điểm trung tâm của xe và điểm trung tâm của hình ảnh
    center_point = (right_point + left_point) // 2
    center_diff = center_point - im_center

    # Tính toán góc lái từ sự khác biệt điểm trung tâm
    steering = center_diff
    steering = min(150, max(-150, steering))
    throttle = 1
    print("steering angle: ",steering)


    if -30 <= steering <= 30:
        # Gửi dữ liệu qua serial
        #ser.write(('F' + "\n").encode())
        print("steering controll: thẳng")
    if -90 <= steering <= -30:
        #ser.write(('Q' + "\n").encode())
        print("steering controll: trái45")
    if -150 <= steering <= -90:
        #ser.write(('L' + "\n").encode())
        print("steering controll: trái")
    if 30 <= steering <= 90:
        #ser.write(('E' + "\n").encode())
        print("steering controll: phải45")
    if 90  <= steering <= 150:
        #ser.write(('R' + "\n").encode())
        print("steering controll: phải ")
    time.sleep(0.05)     
# Từ góc lái, tính toán tốc độ động cơ trái/phải\
    left_motor_speed = 0
    right_motor_speed = 0
    if steering > 0:
        left_motor_speed = 80
        right_motor_speed = throttle * (steering)
    else:
        left_motor_speed = throttle * ( - steering)
        right_motor_speed = 80

    left_motor_speed = int(left_motor_speed )
    right_motor_speed = int(right_motor_speed )
    #print("left_motor_speed: " ,left_motor_speed)
    #print("right_motor_speed: ", right_motor_speed)
    return left_motor_speed, right_motor_speed

# Mở video
cap = cv2.VideoCapture('D:\Picture\Video Projects\line1.mp4')

# Kiểm tra xem video có mở thành công không
if not cap.isOpened():
    print("Error: Không thể mở video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

     # Thay đổi kích thước khung hình về 480x640
    frame = cv2.resize(frame, (640, 480))
    draw = frame.copy()

    left_point, right_point, center, viz_img = find_lane_lines(frame, draw=True)
    left_motor_speed, right_motor_speed = calculate_control_signal(left_point, right_point, center)

    # Hiển thị hình ảnh với các làn đường đã được vẽ
    cv2.imshow("Lane Lines", viz_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
