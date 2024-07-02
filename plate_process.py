import cv2
import numpy as np
import os

# 加载图片 Load picture
def load_image(img_path):
    rawImage = cv2.imread(img_path)
    if rawImage is None:
        raise FileNotFoundError("The image file was not found.")
    return rawImage

# 转换为HSV颜色空间并生成掩码 Convert to HSV color space and generate mask
# 凸显车牌字符 Highlight license plate characters
def filter_color(hsv):
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 30])
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    mask = cv2.bitwise_or(mask_black, mask_white)
    return cv2.bitwise_not(mask)

# 预处理图像 Preprocessed image
def preprocess_image(rawImage):
    # 将原始图像从BGR颜色空间转换为HSV颜色空间
    # Convert the original image from BGR color space to HSV color space
    hsv = cv2.cvtColor(rawImage, cv2.COLOR_BGR2HSV)
    
    # 根据颜色过滤图像以突出显示车牌区域
    # Filter the image based on color to highlight the license plate area
    filtered_img_color = filter_color(hsv)
    
    # 提取HSV图像的V通道（亮度通道）
    # Extract the V channel (brightness) from the HSV image
    v_channel = hsv[:, :, 2]
    
    # 使用CLAHE进行自适应直方图均衡化，增强图像的对比度
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the contrast of the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(v_channel)
    
    # 对图像进行高斯模糊，以减少噪声
    # Apply Gaussian blur to the image to reduce noise
    blurred_img = cv2.GaussianBlur(clahe_img, (3, 3), 0)
    
    # 对图像进行双边滤波，进一步去噪并保持边缘
    # Apply bilateral filter to the image for further noise reduction while preserving edges
    bilateral_filtered_img = cv2.bilateralFilter(blurred_img, 9, 75, 75)
    
    # 使用Sobel算子计算图像的x方向梯度
    # Compute the x-gradient of the image using the Sobel operator
    Sobel_x = cv2.Sobel(bilateral_filtered_img, cv2.CV_16S, 1, 0, ksize=3)
    
    # 将梯度图像转换为8位图像
    # Convert the gradient image to an 8-bit image
    absX = cv2.convertScaleAbs(Sobel_x)
    
    # 对梯度图像进行二值化处理，使用Otsu's方法自动选择阈值
    # Apply binary thresholding to the gradient image using Otsu's method to automatically choose the threshold
    _, threshold_img = cv2.threshold(absX, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 创建水平方向的矩形结构元素并进行图像膨胀操作
    # Create a rectangular structuring element for horizontal dilation
    kernelX = cv2.getStructuringElement(cv2.MORPH_RECT, (36, 5))
    dilated_img = cv2.dilate(threshold_img, kernelX, iterations=1)
    
    # 创建水平方向的矩形结构元素并进行图像腐蚀操作
    # Apply erosion with the same horizontal structuring element
    eroded_img = cv2.erode(dilated_img, kernelX, iterations=1)
    
    # 创建垂直方向的矩形结构元素并进行图像膨胀操作
    # Create a rectangular structuring element for vertical dilation
    kernelY = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 36))
    dilated_img = cv2.dilate(eroded_img, kernelY, iterations=1)
    
    # 进行额外的水平膨胀操作
    # Apply additional horizontal dilation
    dilated_img = cv2.dilate(dilated_img, kernelX, iterations=2)
    
    # 对图像进行中值滤波以去除噪点
    # Apply median blur to remove noise
    filtered_img = cv2.medianBlur(dilated_img, 15)
    
    # 返回预处理后的图像
    # Return the preprocessed image
    return filtered_img


# 轮廓检测和车牌识别
# Contour detection and license plate recognition
def detect_license_plate(filtered_img, rawImage):
    # 找到图像中的所有外部轮廓
    # Find all external contours in the image
    contours, _ = cv2.findContours(filtered_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_rect = None
    max_ratio_diff = float('inf')
    min_area = 10000  # 最小面积, Minimum area

    # 遍历所有轮廓
    # Iterate over all contours
    for item in contours:
        # 获取每个轮廓的边界矩形
        # Get the bounding rectangle for each contour
        rect = cv2.boundingRect(item)
        x, y, width, height = rect
        area = width * height
        
        # 过滤掉面积小于最小面积的轮廓
        # Filter out contours with an area smaller than the minimum area
        if area < min_area:
            continue
        
        # 选择长宽比在合理范围内的轮廓
        # Select contours with a reasonable aspect ratio
        if (width > (height * 2)) and (width < (height * 6)):
            ratio = width / height
            ideal_ratio = 4  # 理想的长宽比, Ideal aspect ratio
            ratio_diff = abs(ratio - ideal_ratio)
            
            # 选择与理想长宽比最接近的轮廓
            # Select the contour closest to the ideal aspect ratio
            if ratio_diff < max_ratio_diff:
                max_ratio_diff = ratio_diff
                best_rect = rect

    # 如果找到合适的矩形
    # If a suitable rectangle is found
    if best_rect is not None:
        x, y, width, height = best_rect
        
        # 扩展边界矩形，以确保包含整个车牌区域
        # Expand the bounding rectangle to ensure it contains the entire license plate area
        x_expanded = max(0, x - 10)
        y_expanded = max(0, y - 10)
        width_expanded = min(rawImage.shape[1], x + width + 10) - x_expanded
        height_expanded = min(rawImage.shape[0], y + height + 10) - y_expanded

        # 从原始图像中裁剪车牌区域
        # Crop the license plate area from the original image
        plate_img = rawImage[y_expanded:y_expanded + height_expanded, x_expanded:x_expanded + width_expanded]
        
        # 如果目录不存在，则创建目录
        # Create the directory if it doesn't exist
        if not os.path.exists('./car_license'):
            os.makedirs('./car_license')
        
        # 将车牌图像保存到指定路径
        # Save the license plate image to the specified path
        cv2.imwrite('./car_license/test.jpg', plate_img)
        
        # 返回车牌图像
        # Return the license plate image
        return plate_img
    else:
        # 如果没有检测到车牌
        # If no license plate is detected
        print("No license plate detected.")
        return None


# 主函数 Main function
def main():
    img_path = '车牌/1.jpg'
    rawImage = load_image(img_path)
    filtered_img = preprocess_image(rawImage)
    plate_img = detect_license_plate(filtered_img, rawImage)
    if plate_img is not None:
        from char_recognize import main as recognize_plate
        svm_result, ocr_result = recognize_plate()
        from database_connection import check_license_plate
        check_license_plate(ocr_result)

if __name__ == "__main__":
    main()
    
    
    
