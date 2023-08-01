import numpy as np

def getPerspectiveTransform(src_points, dst_points):
    if len(src_points) != 4 or len(dst_points) != 4:
        raise ValueError("源点和目标点的数量必须为4个")
    # 构造输入矩阵 A 和输出矩阵 B
    A = np.zeros((8, 8), dtype=np.float32)
    B = np.zeros((8, 1), dtype=np.float32)
    for i in range(4):
        x, y = src_points[i]
        u, v = dst_points[i]
        A[i*2] = [x, y, 1, 0, 0, 0, -x*u, -y*u]
        A[i*2+1] = [0, 0, 0, x, y, 1, -x*v, -y*v]
        B[i*2][0] = u
        B[i*2+1][0] = v
    # 解线性方程组以计算透视变换矩阵
    M = np.linalg.solve(A, B)
    M = np.append(M, 1).reshape(3, 3).astype(np.float32)
    return M

def warpPerspective(image, M, output_size):
    # 获取输入图像的尺寸
    h, w = image.shape[:2]

    # 创建输出图像
    output = np.zeros(output_size, dtype=np.uint8)

    # 计算逆变换矩阵
    M_inv = np.linalg.inv(M)

    # 遍历输出图像的每个像素
    for y in range(output_size[0]):
        for x in range(output_size[1]):
            # 计算输入图像中对应的坐标
            p = np.dot(M_inv, np.array([x, y, 1]))
            p = p / p[2]
            # 检查坐标是否在输入图像范围内
            if p[0] >= 0 and p[0] < w and p[1] >= 0 and p[1] < h:
                # 双线性插值
                x1, y1 = int(p[0]), int(p[1])
                x2, y2 = x1 + 1, y1 + 1

                dx, dy = p[0] - x1, p[1] - y1

                output[y, x] = (1 - dx) * (1 - dy) * image[y1, x1] + \
                               dx * (1 - dy) * image[y1, x2] + \
                               (1 - dx) * dy * image[y2, x1] + \
                               dx * dy * image[y2, x2]

    return output.astype(np.float32)