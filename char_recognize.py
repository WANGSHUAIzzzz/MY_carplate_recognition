import cv2
import numpy as np
import os
import shutil
import joblib
from paddleocr import PaddleOCR
from sklearn import svm
from sklearn.model_selection import train_test_split

# Global configuration
SPLIT_CHARS_DIR = 'car_license/split_chars'
TEMPLATE_DIR = 'chars'
SVM_MODEL_PATH = 'car_license/svm_model.pkl'
PROCESSED_IMG_PATH = 'car_license/processed_plate.jpg'

# Extract HOG features
def extract_hog_features(img):
    hog = cv2.HOGDescriptor((20, 20), (10, 10), (5, 5), (10, 10), 9)
    return hog.compute(img)

# Load template data
def load_template_data(template_dir):
    data, labels = [], []
    for char in os.listdir(template_dir):
        if char in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' and char not in 'IO':
            template_path = os.path.join(template_dir, char)
            for template_file in os.listdir(template_path):
                template_img = cv2.imread(os.path.join(template_path, template_file), cv2.IMREAD_GRAYSCALE)
                if template_img is not None:
                    template_img = cv2.resize(template_img, (20, 20))
                    data.append(extract_hog_features(template_img))
                    labels.append(char)
    return np.array(data, dtype=np.float32), np.array(labels)

# Train SVM model
def train_svm_model():
    data, labels = load_template_data(TEMPLATE_DIR)
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
    model = svm.SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    joblib.dump(model, SVM_MODEL_PATH)
    return model

# Load SVM model
def load_svm_model():
    if not os.path.exists(SVM_MODEL_PATH):
        return train_svm_model()
    return joblib.load(SVM_MODEL_PATH)

# Character segmentation and SVM recognition
def char_recognition_svm(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    ret, img_thre = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.dilate(img_thre, kernel, iterations=1)

    if os.path.exists(SPLIT_CHARS_DIR):
        shutil.rmtree(SPLIT_CHARS_DIR)
    os.makedirs(SPLIT_CHARS_DIR)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    char_rects, padding = [], 2
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 1.0:
                x, y = max(x - padding, 0), max(y - padding, 0)
                w, h = min(w + 2 * padding, closed.shape[1] - x), min(h + 2 * padding, closed.shape[0] - y)
                char_rects.append((x, y, w, h))

    char_rects = sorted(char_rects, key=lambda r: r[0])
    char_count, model = 1, load_svm_model()
    recognized_text = ''
    for rect in char_rects:
        x, y, w, h = rect
        char_img = closed[y:y+h, x:x+w]
        char_img = cv2.erode(char_img, kernel, iterations=1)
        char_img = cv2.resize(char_img, (20, 20))
        hog_features = extract_hog_features(char_img).reshape(1, -1)
        recognized_char = model.predict(hog_features)
        recognized_text += recognized_char[0]
        cv2.imwrite(os.path.join(SPLIT_CHARS_DIR, f'char_{char_count}.jpg'), char_img)
        char_count += 1

    cv2.imwrite(PROCESSED_IMG_PATH, closed)
    return recognized_text, ""

# PaddleOCR recognition
def char_recognition_paddleocr():
    binary_image = cv2.imread(PROCESSED_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    ocr = PaddleOCR(use_angle_cls=True, use_gpu=False, ocr_version='PP-OCRv3')
    result = ocr.ocr(binary_image, cls=True)
    ocr_text = ''.join([res[1][0].replace(" ", "") for line in result for res in line])

    # Revised result
    import re
    match = re.match(r'([A-Z]+)(\d+)', ocr_text)
    if match:
        corrected_text = match.group(1) + match.group(2)
    else:
        corrected_text = ""

    return corrected_text

# Main function
def main():
    img_path = 'car_license/test.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    svm_result, error_message = char_recognition_svm(img_path)
    if error_message:
        return None, None, error_message

    ocr_result = char_recognition_paddleocr()
    return svm_result, ocr_result, ""

if __name__ == "__main__":
    svm_result, ocr_result, error_message = main()
    if error_message:
        print(error_message)
    else:
        print(f"SVM Result: {svm_result}")
        print(f"OCR Result: {ocr_result}")
