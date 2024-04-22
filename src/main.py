import cv2
import numpy
import numpy as np


def filter_close_points(coordinates):  # 滤除密度比较大的点
    filtered_coordinates = []

    if not coordinates:
        return filtered_coordinates

    # 将第一个坐标添加到筛选后的列表中
    filtered_coordinates.append(coordinates[0])

    # 遍历剩余的坐标，只添加与之前所有坐标差值大于等于1的坐标
    for i in range(1, len(coordinates)):
        curr_x, curr_y = coordinates[i]

        # 检查当前坐标与已筛选坐标列表中所有坐标的差值
        valid_coordinate = True
        for filtered_x, filtered_y in filtered_coordinates:
            diff_x = abs(curr_x - filtered_x)
            diff_y = abs(curr_y - filtered_y)
            if diff_x <= 11 and diff_y <= 11:
                valid_coordinate = False
                break

        # 如果与所有已筛选坐标的差值都大于等于1，则将当前坐标添加到筛选后的列表中
        if valid_coordinate:
            filtered_coordinates.append((curr_x, curr_y))

    return filtered_coordinates


def find_min_max_x_coordinates(coordinates):  # 返回四个顶点坐标
    if not coordinates:
        return None, None  # 如果坐标列表为空，返回空值
    # 找出横坐标最小值对应的坐标和最大值对应的坐标
    min_x_coord = min(coordinates, key=lambda coord: coord[0])
    max_x_coord = max(coordinates, key=lambda coord: coord[0])
    min_y_coord = min(coordinates, key=lambda coord: coord[1])
    max_y_coord = max(coordinates, key=lambda coord: coord[1])
    return min_y_coord, min_x_coord, max_y_coord, max_x_coord  # 逆时针


def dilate_contour(contour, scale_factor):
    # 计算轮廓的凸包
    hull = cv2.convexHull(contour)

    # 对凸包进行膨胀操作
    M = cv2.moments(hull)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    hull_scaled = cv2.approxPolyDP(hull, scale_factor, True)
    return hull_scaled


def scale(data, sec_dis):
    """多边形等距缩放
    Args:
        data: 多边形按照逆时针顺序排列的的点集
        sec_dis: 缩放距离

    Returns:
        缩放后的多边形点集
    """
    num = len(data)
    scal_data = []
    for i in range(num):
        x1 = data[i % num][0] - data[(i - 1) % num][0]
        y1 = data[i % num][1] - data[(i - 1) % num][1]
        x2 = data[(i + 1) % num][0] - data[i % num][0]
        y2 = data[(i + 1) % num][1] - data[i % num][1]

        d_A = (x1 ** 2 + y1 ** 2) ** 0.5
        d_B = (x2 ** 2 + y2 ** 2) ** 0.5

        Vec_Cross = (x1 * y2) - (x2 * y1)
        if (d_A * d_B == 0):
            continue
        sin_theta = Vec_Cross / (d_A * d_B)
        if (sin_theta == 0):
            continue
        dv = sec_dis / sin_theta

        v1_x = (dv / d_A) * x1
        v1_y = (dv / d_A) * y1

        v2_x = (dv / d_B) * x2
        v2_y = (dv / d_B) * y2

        PQ_x = v1_x - v2_x
        PQ_y = v1_y - v2_y

        Q_x = data[(i) % num][0] + PQ_x
        Q_y = data[(i) % num][1] + PQ_y
        scal_data.append([int(Q_x), int(Q_y)])
    return scal_data


def detect_black_edges(image_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary_image = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

    # Canny边缘检测
    edges = cv2.Canny(binary_image[1], 50, 150)
    #
    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #
    #
    # # 提取黑色边缘的坐标点
    black_edges_points = []
    #
    for contour in contours:
        # 计算轮廓的外接矩形
        parent_hierarchy = cv2.minEnclosingCircle(contour)
        print(parent_hierarchy)
        if 270 < parent_hierarchy[1] < 300:  # 滤除干扰轮廓
            black_edges_points.extend(contour.reshape(-1, 2))  # 将轮廓点添加到列表中

    # 在原始图像上绘制黑色边缘的坐标点

    # black_edges_points = filter_close_points(black_edges_points)

    # black_edges_points = find_min_max_x_coordinates(black_edges_points)
    # black_edges_points = scale(black_edges_points, -10)
    black_edges_points_scale = scale(black_edges_points, -10)

    for point, points in zip(black_edges_points, black_edges_points_scale):
        cv2.circle(image, tuple(point), 3, (0, 255, 255), 1)  # 绘制圆点
        cv2.circle(image, tuple(points), 3, (0, 255, 0), 1)  # 绘制圆点
        cv2.imshow('Detected Black Edges', image)
        cv2.waitKey(20)
    cv2.waitKey(0)

    return image


# 指定要处理的图像路径
image_path = 'counter.jpg'

# 调用函数进行边缘检测、坐标点提取并绘制
result_image = detect_black_edges(image_path)
#
# # 显示处理后的图片
# cv2.imshow('Detected Black Edges', result_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
