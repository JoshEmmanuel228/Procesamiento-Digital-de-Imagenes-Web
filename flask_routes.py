from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2  # type: ignore
import numpy as np
import base64
import matplotlib
import os
matplotlib.use('Agg')  # Use non-interactive backend

# Type hints for OpenCV functions
cv2.imencode = cv2.imencode  # type: ignore
cv2.imdecode = cv2.imdecode  # type: ignore
cv2.cvtColor = cv2.cvtColor  # type: ignore
cv2.bitwise_not = cv2.bitwise_not  # type: ignore
cv2.threshold = cv2.threshold  # type: ignore
cv2.calcHist = cv2.calcHist  # type: ignore
cv2.Sobel = cv2.Sobel  # type: ignore
cv2.filter2D = cv2.filter2D  # type: ignore
cv2.addWeighted = cv2.addWeighted  # type: ignore
cv2.Laplacian = cv2.Laplacian  # type: ignore
cv2.Scharr = cv2.Scharr  # type: ignore
cv2.normalize = cv2.normalize  # type: ignore
cv2.split = cv2.split  # type: ignore
cv2.merge = cv2.merge  # type: ignore
cv2.LUT = cv2.LUT  # type: ignore
cv2.convertScaleAbs = cv2.convertScaleAbs  # type: ignore
cv2.GaussianBlur = cv2.GaussianBlur  # type: ignore
cv2.medianBlur = cv2.medianBlur  # type: ignore
cv2.bilateralFilter = cv2.bilateralFilter  # type: ignore
cv2.boxFilter = cv2.boxFilter  # type: ignore
cv2.dilate = cv2.dilate  # type: ignore
cv2.erode = cv2.erode  # type: ignore
cv2.morphologyEx = cv2.morphologyEx  # type: ignore
cv2.Canny = cv2.Canny  # type: ignore
cv2.getRotationMatrix2D = cv2.getRotationMatrix2D  # type: ignore
cv2.warpAffine = cv2.warpAffine  # type: ignore
cv2.resize = cv2.resize  # type: ignore
cv2.getPerspectiveTransform = cv2.getPerspectiveTransform  # type: ignore
cv2.warpPerspective = cv2.warpPerspective  # type: ignore
cv2.flip = cv2.flip  # type: ignore
cv2.bitwise_and = cv2.bitwise_and  # type: ignore
cv2.bitwise_or = cv2.bitwise_or  # type: ignore
cv2.bitwise_xor = cv2.bitwise_xor  # type: ignore
cv2.putText = cv2.putText  # type: ignore
cv2.transform = cv2.transform  # type: ignore
cv2.adaptiveThreshold = cv2.adaptiveThreshold  # type: ignore
cv2.distanceTransform = cv2.distanceTransform  # type: ignore
cv2.subtract = cv2.subtract  # type: ignore
cv2.connectedComponents = cv2.connectedComponents  # type: ignore
cv2.watershed = cv2.watershed  # type: ignore
cv2.grabCut = cv2.grabCut  # type: ignore
cv2.cornerHarris = cv2.cornerHarris  # type: ignore
cv2.connectedComponentsWithStats = cv2.connectedComponentsWithStats  # type: ignore
cv2.circle = cv2.circle  # type: ignore
cv2.goodFeaturesToTrack = cv2.goodFeaturesToTrack  # type: ignore
cv2.SimpleBlobDetector_Params = cv2.SimpleBlobDetector_Params  # type: ignore
cv2.SimpleBlobDetector_create = cv2.SimpleBlobDetector_create  # type: ignore
cv2.drawKeypoints = cv2.drawKeypoints  # type: ignore
cv2.SIFT_create = cv2.SIFT_create  # type: ignore
cv2.FlannBasedMatcher = cv2.FlannBasedMatcher  # type: ignore
cv2.drawMatches = cv2.drawMatches  # type: ignore
cv2.ORB_create = cv2.ORB_create  # type: ignore
cv2.BFMatcher = cv2.BFMatcher  # type: ignore
cv2.CascadeClassifier = cv2.CascadeClassifier  # type: ignore
cv2.rectangle = cv2.rectangle  # type: ignore
cv2.findContours = cv2.findContours  # type: ignore
cv2.contourArea = cv2.contourArea  # type: ignore
cv2.boundingRect = cv2.boundingRect  # type: ignore
cv2.drawContours = cv2.drawContours  # type: ignore
cv2.moments = cv2.moments  # type: ignore

# Initialize Flask app
app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for image data
image_data = {
    'image1': None,
    'image2': None,
    'original_image1': None,
    'original_image2': None,
    'processed_image': None,
    'active_image': None
}

# Root route to serve the main page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Serve static files
@app.route('/<path:path>')
def serve_static(path):
    if path.endswith('.css'):
        return send_from_directory('.', path, mimetype='text/css')
    elif path.endswith('.js'):
        return send_from_directory('.', path, mimetype='application/javascript')
    return send_from_directory('.', path)

# Helper functions for image conversion
def cv2_to_base64(img):
    try:
        # Ensure image is in the correct format
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        
        # Encode image to PNG
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error converting image to base64: {str(e)}")
        return None

def base64_to_cv2(base64_string):
    try:
        # Decode base64 string
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        return None

def add_salt_and_pepper_noise(image, ratio, amount):
    """
    Add salt and pepper noise to an image.
    
    Args:
        image: Input image
        ratio: Ratio of salt to pepper (0-1)
        amount: Amount of noise to add (0-1)
    
    Returns:
        Noisy image
    """
    noisy = image.copy()
    num_salt = np.ceil(amount * image.size * ratio)
    num_pepper = np.ceil(amount * image.size * (1.0 - ratio))
    
    # Add Salt noise
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy[tuple(coords)] = 255
    
    # Add Pepper noise
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy[tuple(coords)] = 0
    
    return noisy

def add_gaussian_noise(image, mean, std_dev):
    """
    Add Gaussian noise to an image.
    
    Args:
        image: Input image
        mean: Mean of the Gaussian noise
        std_dev: Standard deviation of the Gaussian noise
    
    Returns:
        Noisy image
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, std_dev, (row, col, ch))
    noisy = image + gauss
    return np.clip(noisy, 0, 255).astype(np.uint8)

def update_processed_image(processed_img):
    """Helper function to update the correct image based on which one is active"""
    if image_data['active_image'] is image_data['image1']:
        image_data['image1'] = processed_img.copy()
        image_data['active_image'] = image_data['image1']
    elif image_data['active_image'] is image_data['image2']:
        image_data['image2'] = processed_img.copy()
        image_data['active_image'] = image_data['image2']
    image_data['processed_image'] = processed_img.copy()

# Flask routes
@app.route('/load_image1', methods=['POST'])
def load_image1():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            
        # Read and process the image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return jsonify({'error': 'Error al procesar la imagen'}), 400
            
        # Store both the image and its original copy
        image_data['image1'] = img.copy()
        image_data['original_image1'] = img.copy()
        image_data['active_image'] = image_data['image1']
        image_data['processed_image'] = image_data['image1'].copy()
        
        # Convert to base64 for response
        img_base64 = cv2_to_base64(img)
        return jsonify({
            'message': 'Imagen 1 cargada exitosamente',
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_image2', methods=['POST'])
def load_image2():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No se proporcionó ninguna imagen'}), 400
            
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
            
        # Read and process the image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            return jsonify({'error': 'Error al procesar la imagen'}), 400
            
        # Store both the image and its original copy
        image_data['image2'] = img.copy()
        image_data['original_image2'] = img.copy()
        image_data['active_image'] = image_data['image2']
        image_data['processed_image'] = image_data['image2'].copy()
        
        # Convert to base64 for response
        img_base64 = cv2_to_base64(img)
        return jsonify({
            'message': 'Imagen 2 cargada exitosamente',
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/select_image', methods=['POST'])
def select_image():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400
            
        # Set the selected image as active
        image_data['active_image'] = image_data[image_key]
        image_data['processed_image'] = image_data[image_key].copy()
        
        # Convert to base64 for response
        img_base64 = cv2_to_base64(image_data['active_image'])
        return jsonify({
            'message': f'Imagen {image_number} seleccionada',
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/restore_image', methods=['POST'])
def restore_image():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        original_key = f'original_{image_key}'
        if image_data[image_key] is None or image_data[original_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400
            
        # Restore the image to its original state
        image_data[image_key] = image_data[original_key].copy()
        image_data['active_image'] = image_data[image_key]
        image_data['processed_image'] = image_data[image_key].copy()
        
        # Convert to base64 for response
        img_base64 = cv2_to_base64(image_data['active_image'])
        return jsonify({
            'message': f'Imagen {image_number} restaurada',
            'image': img_base64
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_gray', methods=['POST'])
def apply_gray():
    try:
        data = request.get_json()
        invert = data.get('invert', False)
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        # Convert to grayscale
        if img.ndim == 3:
            if img.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:  # RGB
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Invert if requested
        if invert:
            gray = cv2.bitwise_not(gray)

        # Update the correct image
        image_data[image_key] = gray.copy()
        image_data['active_image'] = gray.copy()
        image_data['processed_image'] = gray.copy()

        # Convert to base64 for response
        img_base64 = cv2_to_base64(gray)
        return jsonify({
            'message': 'Escala de grises aplicada exitosamente',
            'image': img_base64,
            'inverted': invert
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_threshold', methods=['POST'])
def apply_threshold():
    try:
        data = request.get_json()
        threshold_value = data.get('threshold', 127)
        invert = data.get('invert', False)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Convert to grayscale if needed
        if img.ndim == 3:
            if img.shape[2] == 4:  # RGBA
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:  # RGB
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply threshold
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        _, thresholded = cv2.threshold(gray, threshold_value, 255, thresh_type)

        # Update the correct image
        update_processed_image(thresholded)

        # Convert to base64 for response
        img_base64 = cv2_to_base64(thresholded)
        return jsonify({
            'message': 'Threshold applied successfully',
            'image': img_base64,
            'threshold': threshold_value,
            'inverted': invert
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_noise', methods=['POST'])
def apply_noise():
    try:
        data = request.get_json()
        noise_type = data.get('type', 'gaussian')
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if noise_type == 'salt_pepper':
            amount = data.get('amount', 0.05)
            ratio = data.get('ratio', 0.5)
            noisy = add_salt_and_pepper_noise(img, ratio, amount)
        else:  # gaussian
            std_dev = data.get('std_dev', 25)
            mean = data.get('mean', 0)
            noisy = add_gaussian_noise(img, mean, std_dev)

        image_data['processed_image'] = noisy

        img_base64 = cv2_to_base64(noisy)
        return jsonify({
            'message': f'{noise_type.capitalize()} noise applied successfully',
            'image': img_base64,
            'noise_type': noise_type
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_histogram', methods=['POST'])
def get_histogram():
    try:
        data = request.get_json()
        source = data.get('source', 'active')
        
        img = None
        if source == 'active':
            img = image_data['active_image']
        else:
            img = image_data['processed_image']

        if img is None:
            return jsonify({'error': 'No hay imagen disponible para calcular el histograma'}), 400

        histograms = {}
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            else:
                img_bgr = img

            histograms['blue'] = cv2.calcHist([img_bgr], [0], None, [256], [0, 256]).flatten().tolist()
            histograms['green'] = cv2.calcHist([img_bgr], [1], None, [256], [0, 256]).flatten().tolist()
            histograms['red'] = cv2.calcHist([img_bgr], [2], None, [256], [0, 256]).flatten().tolist()

            # Re-use the correct conversion for BGRA
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            histograms['luminance'] = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten().tolist()
        else:
            histograms['luminance'] = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten().tolist()

        return jsonify({
            'message': 'Histogram calculated successfully',
            'histograms': histograms
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_convolution', methods=['POST'])
def apply_convolution():
    try:
        data = request.get_json()
        filter_type = data.get('type', 'sobel')
        direction = data.get('direction', 'both')
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if filter_type == 'sobel':
            if direction == 'x':
                filtered = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            elif direction == 'y':
                filtered = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            else:  # both
                filtered = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        elif filter_type == 'prewitt':
            kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            if direction == 'x':
                filtered = cv2.filter2D(gray, -1, kernel_x)
            elif direction == 'y':
                filtered = cv2.filter2D(gray, -1, kernel_y)
            else:  # both
                filtered_x = cv2.filter2D(gray, -1, kernel_x)
                filtered_y = cv2.filter2D(gray, -1, kernel_y)
                filtered = cv2.addWeighted(filtered_x, 0.5, filtered_y, 0.5, 0)
        elif filter_type == 'laplacian':
            filtered = cv2.Laplacian(gray, cv2.CV_64F)
        elif filter_type == 'scharr':
            if direction == 'x':
                filtered = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
            elif direction == 'y':
                filtered = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
            else:  # both
                filtered_x = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
                filtered_y = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
                filtered = cv2.addWeighted(filtered_x, 0.5, filtered_y, 0.5, 0)
        else:
            return jsonify({'error': 'Invalid filter type'}), 400

        # Normalize the filtered image
        filtered = cv2.normalize(filtered, None, 0, 255, cv2.NORM_MINMAX)
        filtered = filtered.astype(np.uint8)

        image_data['processed_image'] = filtered

        img_base64 = cv2_to_base64(filtered)
        return jsonify({
            'message': f'{filter_type.capitalize()} filter applied successfully',
            'image': img_base64,
            'filter_type': filter_type,
            'direction': direction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_gamma', methods=['POST'])
def apply_gamma():
    try:
        data = request.get_json()
        gamma = data.get('gamma', 1.0)
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        # Build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        
        # Apply gamma correction using the lookup table
        if img.ndim == 3:
            if img.shape[2] == 4:
                # Split the image into channels
                b, g, r, a = cv2.split(img)
                # Apply gamma correction to each channel
                b = cv2.LUT(b, table)
                g = cv2.LUT(g, table)
                r = cv2.LUT(r, table)
                # Merge the channels back
                corrected = cv2.merge([b, g, r, a])
            else:
                # Split the image into channels
                b, g, r = cv2.split(img)
                # Apply gamma correction to each channel
                b = cv2.LUT(b, table)
                g = cv2.LUT(g, table)
                r = cv2.LUT(r, table)
                # Merge the channels back
                corrected = cv2.merge([b, g, r])
        else:
            corrected = cv2.LUT(img, table)

        # Update the correct image
        image_data[image_key] = corrected.copy()
        image_data['active_image'] = corrected.copy()
        image_data['processed_image'] = corrected.copy()

        # Convert to base64 for response
        img_base64 = cv2_to_base64(corrected)
        return jsonify({
            'message': 'Corrección gamma aplicada exitosamente',
            'image': img_base64,
            'gamma': gamma
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_contrast', methods=['POST'])
def apply_contrast():
    try:
        data = request.get_json()
        alpha = data.get('alpha', 1.0)  # Contrast control
        beta = data.get('beta', 0)      # Brightness control
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Apply contrast and brightness
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        image_data['processed_image'] = adjusted

        img_base64 = cv2_to_base64(adjusted)
        return jsonify({
            'message': 'Contrast and brightness adjusted successfully',
            'image': img_base64,
            'alpha': alpha,
            'beta': beta
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_smoothing', methods=['POST'])
def apply_smoothing():
    try:
        data = request.get_json()
        filter_type = data.get('type', 'gaussian')
        kernel_size = data.get('kernel_size', 5)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if filter_type == 'gaussian':
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        elif filter_type == 'median':
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = cv2.medianBlur(img, kernel_size)
        elif filter_type == 'bilateral':
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = cv2.bilateralFilter(img, kernel_size, 75, 75)
        elif filter_type == 'box':
            smoothed = cv2.boxFilter(img, -1, (kernel_size, kernel_size))
        else:
            return jsonify({'error': 'Invalid filter type'}), 400

        image_data['processed_image'] = smoothed

        img_base64 = cv2_to_base64(smoothed)
        return jsonify({
            'message': f'{filter_type.capitalize()} smoothing applied successfully',
            'image': img_base64,
            'filter_type': filter_type,
            'kernel_size': kernel_size
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_morphology', methods=['POST'])
def apply_morphology():
    try:
        data = request.get_json()
        operation = data.get('operation', 'dilate')
        kernel_size = data.get('kernel_size', 5)
        iterations = data.get('iterations', 1)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if operation == 'dilate':
            result = cv2.dilate(img, kernel, iterations=iterations)
        elif operation == 'erode':
            result = cv2.erode(img, kernel, iterations=iterations)
        elif operation == 'open':
            result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'gradient':
            result = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        elif operation == 'tophat':
            result = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
        elif operation == 'blackhat':
            result = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
        else:
            return jsonify({'error': 'Invalid operation'}), 400

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'{operation.capitalize()} operation applied successfully',
            'image': img_base64,
            'operation': operation,
            'kernel_size': kernel_size,
            'iterations': iterations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_edge_detection', methods=['POST'])
def apply_edge_detection():
    try:
        data = request.get_json()
        method = data.get('method', 'canny')
        threshold1 = data.get('threshold1', 100)
        threshold2 = data.get('threshold2', 200)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if method == 'canny':
            edges = cv2.Canny(gray, threshold1, threshold2)
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        elif method == 'prewitt':
            kernelx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            kernely = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            prewittx = cv2.filter2D(gray, -1, kernelx)
            prewitty = cv2.filter2D(gray, -1, kernely)
            edges = np.sqrt(prewittx**2 + prewitty**2)
            edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            return jsonify({'error': 'Invalid method'}), 400

        image_data['processed_image'] = edges

        img_base64 = cv2_to_base64(edges)
        return jsonify({
            'message': f'{method.capitalize()} edge detection applied successfully',
            'image': img_base64,
            'method': method,
            'threshold1': threshold1,
            'threshold2': threshold2
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_rotation', methods=['POST'])
def apply_rotation():
    try:
        data = request.get_json()
        angle = data.get('angle', 0)
        scale = data.get('scale', 1.0)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Calculate rotation matrix
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Apply rotation
        rotated = cv2.warpAffine(img, rotation_matrix, (width, height))

        image_data['processed_image'] = rotated

        img_base64 = cv2_to_base64(rotated)
        return jsonify({
            'message': 'Rotation applied successfully',
            'image': img_base64,
            'angle': angle,
            'scale': scale
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_scaling', methods=['POST'])
def apply_scaling():
    try:
        data = request.get_json()
        scale_x = data.get('scale_x', 1.0)
        scale_y = data.get('scale_y', 1.0)
        interpolation = data.get('interpolation', 'linear')
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Get image dimensions
        height, width = img.shape[:2]
        new_width = int(width * scale_x)
        new_height = int(height * scale_y)
        
        # Select interpolation method
        if interpolation == 'nearest':
            inter = cv2.INTER_NEAREST
        elif interpolation == 'linear':
            inter = cv2.INTER_LINEAR
        elif interpolation == 'cubic':
            inter = cv2.INTER_CUBIC
        elif interpolation == 'lanczos':
            inter = cv2.INTER_LANCZOS4
        else:
            return jsonify({'error': 'Invalid interpolation method'}), 400
        
        # Apply scaling
        scaled = cv2.resize(img, (new_width, new_height), interpolation=inter)

        image_data['processed_image'] = scaled

        img_base64 = cv2_to_base64(scaled)
        return jsonify({
            'message': 'Scaling applied successfully',
            'image': img_base64,
            'scale_x': scale_x,
            'scale_y': scale_y,
            'interpolation': interpolation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_perspective', methods=['POST'])
def apply_perspective():
    try:
        data = request.get_json()
        points = data.get('points', None)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400
        
        if points is None or len(points) != 4:
            return jsonify({'error': 'Invalid points for perspective transformation'}), 400

        img = image_data['active_image'].copy()
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Convert points to numpy array
        src_points = np.array(points, dtype=np.float32)
        
        # Define destination points (corners of the image)
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(img, matrix, (width, height))

        image_data['processed_image'] = warped

        img_base64 = cv2_to_base64(warped)
        return jsonify({
            'message': 'Perspective transformation applied successfully',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_flip', methods=['POST'])
def apply_flip():
    try:
        data = request.get_json()
        flip_code = data.get('flip_code', 1)  # 0: vertical, 1: horizontal, -1: both
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Apply flip
        flipped = cv2.flip(img, flip_code)

        image_data['processed_image'] = flipped

        img_base64 = cv2_to_base64(flipped)
        return jsonify({
            'message': 'Flip applied successfully',
            'image': img_base64,
            'flip_code': flip_code
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_blend', methods=['POST'])
def apply_blend():
    try:
        data = request.get_json()
        alpha = data.get('alpha', 0.5)
        
        if image_data['image1'] is None or image_data['image2'] is None:
            return jsonify({'error': 'Both images must be loaded'}), 400

        img1 = image_data['image1'].copy()
        img2 = image_data['image2'].copy()
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
        image_data['processed_image'] = blended

        img_base64 = cv2_to_base64(blended)
        return jsonify({
            'message': 'Blending applied successfully',
            'image': img_base64,
            'alpha': alpha
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_bitwise', methods=['POST'])
def apply_bitwise():
    try:
        data = request.get_json()
        operation = data.get('operation', 'and')
        
        if image_data['image1'] is None or image_data['image2'] is None:
            return jsonify({'error': 'Both images must be loaded'}), 400

        img1 = image_data['image1'].copy()
        img2 = image_data['image2'].copy()
        
        # Validar tipos de datos
        if img1.dtype != np.uint8:
            img1 = np.clip(img1, 0, 255).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = np.clip(img2, 0, 255).astype(np.uint8)
        
        # Manejar diferentes dimensiones
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        # Manejar diferentes canales
        if img1.ndim != img2.ndim:
            if img1.ndim == 2:  # img1 es escala de grises
                img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            if img2.ndim == 2:  # img2 es escala de grises
                img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Manejar BGRA
        if img1.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
        if img2.shape[2] == 4:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
        
        if operation == 'and':
            result = cv2.bitwise_and(img1, img2)
        elif operation == 'or':
            result = cv2.bitwise_or(img1, img2)
        elif operation == 'xor':
            result = cv2.bitwise_xor(img1, img2)
        elif operation == 'not':
            result = cv2.bitwise_not(img1)
        else:
            return jsonify({'error': 'Invalid operation'}), 400

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'Bitwise {operation} operation applied successfully',
            'image': img_base64,
            'operation': operation
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_watermark', methods=['POST'])
def apply_watermark():
    try:
        data = request.get_json()
        text = data.get('text', 'Watermark')
        position = data.get('position', [10, 30])
        font_scale = data.get('font_scale', 1.0)
        color = data.get('color', [255, 255, 255])
        thickness = data.get('thickness', 2)
        alpha = data.get('alpha', 0.5)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        overlay = img.copy()
        
        cv2.putText(overlay, text, tuple(position), cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, color, thickness)
        
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        image_data['processed_image'] = img

        img_base64 = cv2_to_base64(img)
        return jsonify({
            'message': 'Watermark applied successfully',
            'image': img_base64,
            'text': text,
            'position': position,
            'font_scale': font_scale,
            'color': color,
            'thickness': thickness,
            'alpha': alpha
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_sepia', methods=['POST'])
def apply_sepia():
    try:
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        img_float = img.astype(np.float32) / 255.0
        
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],
            [0.349, 0.686, 0.168],
            [0.272, 0.534, 0.131]
        ])
        
        sepia = cv2.transform(img_float, sepia_matrix)
        sepia = np.clip(sepia, 0, 1)
        sepia = (sepia * 255).astype(np.uint8)
        image_data['processed_image'] = sepia

        img_base64 = cv2_to_base64(sepia)
        return jsonify({
            'message': 'Sepia effect applied successfully',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_threshold_adaptive', methods=['POST'])
def apply_threshold_adaptive():
    try:
        data = request.get_json()
        method = data.get('method', 'gaussian')
        block_size = data.get('block_size', 11)
        c = data.get('c', 2)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Ensure block size is odd
        if block_size % 2 == 0:
            block_size += 1

        if method == 'gaussian':
            thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, block_size, c)
        else:  # mean
            thresholded = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                             cv2.THRESH_BINARY, block_size, c)

        image_data['processed_image'] = thresholded

        img_base64 = cv2_to_base64(thresholded)
        return jsonify({
            'message': f'Adaptive threshold ({method}) applied successfully',
            'image': img_base64,
            'method': method,
            'block_size': block_size,
            'c': c
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_otsu', methods=['POST'])
def apply_otsu():
    try:
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Otsu's thresholding
        _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        image_data['processed_image'] = thresholded

        img_base64 = cv2_to_base64(thresholded)
        return jsonify({
            'message': "Otsu's thresholding applied successfully",
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_watershed', methods=['POST'])
def apply_watershed():
    try:
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)

        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1

        # Now, mark the region of unknown with zero
        markers[unknown == 255] = 0

        # Apply watershed
        markers = cv2.watershed(img, markers)

        # Create result image
        result = img.copy()
        result[markers == -1] = [255, 0, 0]  # Mark boundaries in red

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'Watershed segmentation applied successfully',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_grabcut', methods=['POST'])
def apply_grabcut():
    try:
        data = request.get_json()
        rect = data.get('rect', None)  # [x, y, width, height]
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400
        
        if rect is None:
            return jsonify({'error': 'Rectangle coordinates required'}), 400

        img = image_data['active_image'].copy()
        
        # Create mask
        mask = np.zeros(img.shape[:2], np.uint8)
        
        # Create temporary arrays
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Convert rect to tuple
        rect = tuple(rect)
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where 0 and 2 are background, 1 and 3 are foreground
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask to image
        result = img * mask2[:, :, np.newaxis]

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'GrabCut segmentation applied successfully',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_corner_detection', methods=['POST'])
def apply_corner_detection():
    try:
        data = request.get_json()
        method = data.get('method', 'harris')
        block_size = data.get('block_size', 2)
        ksize = data.get('ksize', 3)
        k = data.get('k', 0.04)
        threshold = data.get('threshold', 0.01)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if method == 'harris':
            # Ensure ksize is odd
            if ksize % 2 == 0:
                ksize += 1
            
            # Apply Harris corner detection
            corners = cv2.cornerHarris(gray, block_size, ksize, k)
            
            # Normalize and threshold
            corners = cv2.normalize(corners, None, 0, 255, cv2.NORM_MINMAX)
            corners = np.uint8(corners)
            
            # Threshold for an optimal value
            _, corners = cv2.threshold(corners, threshold * corners.max(), 255, 0)
            
            # Find centroids
            corners = cv2.connectedComponentsWithStats(corners, 8, cv2.CV_32S)
            
            # Draw corners
            result = img.copy()
            for i in range(1, corners[0]):
                x, y = corners[3][i]
                cv2.circle(result, (int(x), int(y)), 3, (0, 255, 0), -1)
        
        elif method == 'shi_tomasi':
            # Ensure ksize is odd
            if ksize % 2 == 0:
                ksize += 1
            
            # Apply Shi-Tomasi corner detection
            corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)
            corners = np.int0(corners)
            
            # Draw corners
            result = img.copy()
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (x, y), 3, (0, 255, 0), -1)
        
        else:
            return jsonify({'error': 'Invalid method'}), 400

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'{method.capitalize()} corner detection applied successfully',
            'image': img_base64,
            'method': method
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_blob_detection', methods=['POST'])
def apply_blob_detection():
    try:
        data = request.get_json()
        min_area = data.get('min_area', 100)
        max_area = data.get('max_area', 1000)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Set up the SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by Area
        params.filterByArea = True
        params.minArea = min_area
        params.maxArea = max_area
        
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
        
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.01
        
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Draw detected blobs as red circles
        result = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'Blob detection applied successfully',
            'image': img_base64,
            'num_blobs': len(keypoints)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_feature_matching', methods=['POST'])
def apply_feature_matching():
    try:
        data = request.get_json()
        method = data.get('method', 'sift')
        max_features = data.get('max_features', 100)
        
        if image_data['image1'] is None or image_data['image2'] is None:
            return jsonify({'error': 'Both images must be loaded'}), 400

        img1 = image_data['image1'].copy()
        img2 = image_data['image2'].copy()
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        if img1.ndim == 3:
            if img1.shape[2] == 4:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
            else:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1.copy()
            gray2 = img2.copy()

        if method == 'sift':
            # Initialize SIFT detector
            sift = cv2.SIFT_create(nfeatures=max_features)
            
            # Find keypoints and descriptors
            kp1, des1 = sift.detectAndCompute(gray1, None)
            kp2, des2 = sift.detectAndCompute(gray2, None)
            
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            # Create FLANN matcher
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find matches
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            
            # Draw matches
            result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        elif method == 'orb':
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=max_features)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            # Create BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            # Find matches
            matches = bf.match(des1, des2)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Draw matches
            result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        else:
            return jsonify({'error': 'Invalid method'}), 400

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'{method.upper()} feature matching applied successfully',
            'image': img_base64,
            'method': method,
            'num_matches': len(good_matches) if method == 'sift' else len(matches)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_face_detection', methods=['POST'])
def apply_face_detection():
    try:
        data = request.get_json()
        method = data.get('method', 'haar')
        min_neighbors = data.get('min_neighbors', 5)
        scale_factor = data.get('scale_factor', 1.1)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        if method == 'haar':
            # Load Haar cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            
            # Draw rectangle around faces
            result = img.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        elif method == 'dnn':
            # Load pre-trained model
            model = cv2.dnn.readNetFromCaffe(
                'deploy.prototxt',
                'res10_300x300_ssd_iter_140000.caffemodel'
            )
            
            # Prepare image for DNN
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123))
            
            # Set input and forward pass
            model.setInput(blob)
            detections = model.forward()
            
            # Draw rectangles around faces
            result = img.copy()
            h, w = img.shape[:2]
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(result, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        else:
            return jsonify({'error': 'Invalid method'}), 400

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'{method.capitalize()} face detection applied successfully',
            'image': img_base64,
            'method': method,
            'num_faces': len(faces) if method == 'haar' else detections.shape[2]
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_object_detection', methods=['POST'])
def apply_object_detection():
    try:
        data = request.get_json()
        confidence_threshold = data.get('confidence_threshold', 0.5)
        
        if image_data['active_image'] is None:
            return jsonify({'error': 'No active image'}), 400

        img = image_data['active_image'].copy()
        
        # Load YOLO model
        net = cv2.dnn.readNetFromDarknet(
            'yolov3.cfg',
            'yolov3.weights'
        )
        
        # Get output layer names
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        
        # Load COCO class names
        with open('coco.names', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        
        # Prepare image for YOLO
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        # Set input and forward pass
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Process detections
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        
        # Draw boxes and labels
        result = img.copy()
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                
                # Draw rectangle and label
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, f'{label} {confidence:.2f}', (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'Object detection applied successfully',
            'image': img_base64,
            'num_objects': len(indexes)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_motion_detection', methods=['POST'])
def apply_motion_detection():
    try:
        data = request.get_json()
        threshold = data.get('threshold', 25)
        min_area = data.get('min_area', 500)
        
        if image_data['image1'] is None or image_data['image2'] is None:
            return jsonify({'error': 'Both images must be loaded'}), 400

        img1 = image_data['image1'].copy()
        img2 = image_data['image2'].copy()
        
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        
        if img1.ndim == 3:
            if img1.shape[2] == 4:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2GRAY)
            else:
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = img1.copy()
            gray2 = img2.copy()

        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Apply threshold
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours
        result = img1.copy()
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'Motion detection applied successfully',
            'image': img_base64,
            'num_motions': len(contours)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_color_space', methods=['POST'])
def apply_color_space():
    try:
        data = request.get_json()
        space = data.get('space')
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if space == 'LAB':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        elif space == 'YCrCb':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        elif space == 'HLS':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        elif space == 'YUV':
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        else:
            return jsonify({'error': 'Espacio de color inválido'}), 400

        image_data['processed_image'] = converted

        img_base64 = cv2_to_base64(converted)
        return jsonify({
            'message': f'Espacio de color {space} aplicado exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_channel', methods=['POST'])
def show_channel():
    try:
        data = request.get_json()
        channel = data.get('channel')
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            if channel == 'R':
                channel_img = img[:, :, 2]
            elif channel == 'G':
                channel_img = img[:, :, 1]
            elif channel == 'B':
                channel_img = img[:, :, 0]
            else:
                return jsonify({'error': 'Canal inválido'}), 400
        else:
            return jsonify({'error': 'La imagen debe ser a color'}), 400

        image_data['processed_image'] = channel_img

        img_base64 = cv2_to_base64(channel_img)
        return jsonify({
            'message': f'Canal {channel} mostrado exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_binary', methods=['POST'])
def show_binary():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_data['processed_image'] = binary

        img_base64 = cv2_to_base64(binary)
        return jsonify({
            'message': 'Imagen binarizada mostrada exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_vec4', methods=['POST'])
def show_vec4():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=4)
        
        # Create a color image for visualization
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background color
        
        colored_labels = colors[labels]
        image_data['processed_image'] = colored_labels

        img_base64 = cv2_to_base64(colored_labels)
        return jsonify({
            'message': 'Etiquetas vecindad-4 mostradas exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_vec8', methods=['POST'])
def show_vec8():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        
        # Create a color image for visualization
        colors = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
        colors[0] = [0, 0, 0]  # Background color
        
        colored_labels = colors[labels]
        image_data['processed_image'] = colored_labels

        img_base64 = cv2_to_base64(colored_labels)
        return jsonify({
            'message': 'Etiquetas vecindad-8 mostradas exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/show_contours', methods=['POST'])
def show_contours():
    try:
        data = request.get_json()
        image_number = data.get('image_number')
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Número de imagen inválido'}), 400
            
        image_key = f'image{image_number}'
        if image_data[image_key] is None:
            return jsonify({'error': f'No hay imagen {image_number} cargada'}), 400

        img = image_data[image_key].copy()
        
        if img.ndim == 3:
            if img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and numbers
        result = img.copy()
        for i, contour in enumerate(contours):
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(result, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        image_data['processed_image'] = result

        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': 'Contornos mostrados exitosamente',
            'image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_arithmetic', methods=['POST'])
def apply_arithmetic():
    try:
        data = request.get_json()
        operation = data.get('operation')
        value = float(data.get('value', 0))
        image_number = int(data.get('image_number', 1))
        
        if image_number not in [1, 2]:
            return jsonify({'error': 'Invalid image number'}), 400
            
        img = image_data[f'image{image_number}'].copy()
        if img is None:
            return jsonify({'error': f'Image {image_number} not loaded'}), 400
            
        # Convert to float32 for arithmetic operations
        img_float = img.astype(np.float32)
        
        # Apply the arithmetic operation
        if operation == 'add':
            result = np.clip(img_float + value, 0, 255).astype(np.uint8)
        elif operation == 'subtract':
            result = np.clip(img_float - value, 0, 255).astype(np.uint8)
        elif operation == 'multiply':
            result = np.clip(img_float * value, 0, 255).astype(np.uint8)
        elif operation == 'divide':
            if abs(value) < 1e-6:
                return jsonify({'error': 'Division by zero'}), 400
            result = np.clip(img_float / value, 0, 255).astype(np.uint8)
        else:
            return jsonify({'error': 'Invalid operation'}), 400
            
        # Update the processed image
        image_data['processed_image'] = result
        
        # Convert to base64 for response
        img_base64 = cv2_to_base64(result)
        return jsonify({
            'message': f'Arithmetic {operation} operation applied successfully',
            'image': img_base64,
            'operation': operation,
            'value': value
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/apply_logical', methods=['POST'])
def apply_logical():
    try:
        data = request.get_json()
        operation = data.get('operation', 'and').upper() # Convert to uppercase for consistent comparison

        # Validar que las imágenes necesarias estén cargadas
        if operation == 'NOT':
            if image_data['image1'] is None:
                return jsonify({'error': 'No hay imagen 1 cargada para aplicar NOT'}), 400
            img1 = image_data['image1'].copy()
            img2 = None # NOT operation is unary
        elif operation in ['AND', 'OR', 'XOR']:
            if image_data['image1'] is None or image_data['image2'] is None:
                return jsonify({'error': f'Ambas imágenes (Imagen 1 y Imagen 2) deben estar cargadas para la operación binaria {operation}'}), 400
            img1 = image_data['image1'].copy() # Usar copia de las imágenes originales cargadas
            img2 = image_data['image2'].copy()
        else:
            return jsonify({'error': f'Operación lógica inválida: {operation}'}), 400

        # --- Validar y convertir tipos de datos a uint8 ---
        # Las operaciones bit a bit de OpenCV requieren uint8
        if img1.dtype != np.uint8:
            try:
                img1 = np.clip(img1, 0, 255).astype(np.uint8)
                print("Imagen 1 convertida a uint8.")
            except Exception as e:
                return jsonify({'error': f'Error al convertir Imagen 1 a uint8: {str(e)}'}), 400

        if img2 is not None and img2.dtype != np.uint8:
            try:
                img2 = np.clip(img2, 0, 255).astype(np.uint8)
                print("Imagen 2 convertida a uint8.")
            except Exception as e:
                return jsonify({'error': f'Error al convertir Imagen 2 a uint8: {str(e)}'}), 400

        # --- Para operaciones binarias, manejar diferencias de dimensiones y canales ---
        if operation in ['AND', 'OR', 'XOR']:
            h1, w1 = img1.shape[:2]
            h2, w2 = img2.shape[:2]

            # Redimensionar Imagen 2 para que coincida con Imagen 1 si las dimensiones difieren
            # Podemos añadir una opción para redimensionar al máximo si se prefiere en el futuro
            if (h1, w1) != (h2, w2):
                print(f"Dimensiones de imagen diferentes: Imagen 1 ({w1}x{h1}), Imagen 2 ({w2}x{h2}). Redimensionando Imagen 2 para coincidir con Imagen 1.")
                try:
                    # Usar INTER_AREA para reducción e INTER_LINEAR para ampliación
                    interpolation = cv2.INTER_AREA if (w2*h2 > w1*h1) else cv2.INTER_LINEAR
                    img2 = cv2.resize(img2, (w1, h1), interpolation=interpolation)
                    print(f"Imagen 2 redimensionada a {img2.shape[1]}x{img2.shape[0]}.")
                except Exception as e:
                     return jsonify({'error': f'Error al redimensionar Imagen 2 para operación binaria: {str(e)}'}), 400

            # Manejar diferencias de canales (Convertir a BGR si es necesario para homogeneizar)
            # Las operaciones bit a bit requieren el mismo número de canales o que una sea escala de grises y la otra no.
            # Un enfoque robusto es convertir ambas a BGR si son de 3 o 4 canales, o mantenerlas como escala de grises si ambas lo son.
            # Si una es escala de grises y la otra color, intentamos convertir la de escala de grises a BGR.

            ndim1, ndim2 = img1.ndim, img2.ndim
            channels1 = img1.shape[2] if ndim1 == 3 else 1
            channels2 = img2.shape[2] if ndim2 == 3 else 1

            # Convertir BGRA a BGR si aplica
            if ndim1 == 3 and channels1 == 4:
                try:
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGRA2BGR)
                    channels1 = 3
                    print("Imagen 1 BGRA convertida a BGR.")
                except Exception as e:
                     return jsonify({'error': f'Error al convertir Imagen 1 BGRA a BGR: {str(e)}'}), 400
            if ndim2 == 3 and channels2 == 4:
                try:
                    img2 = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)
                    channels2 = 3
                    print("Imagen 2 BGRA convertida a BGR.")
                except Exception as e:
                     return jsonify({'error': f'Error al convertir Imagen 2 BGRA a BGR: {str(e)}'}), 400

            # Si después de manejar BGRA, los canales aún no coinciden o los tipos de dato no son iguales (uint8 ya se manejó),
            # intentar homogeneizar a BGR si alguna es a color.
            if channels1 != channels2 or img1.dtype != img2.dtype: # Check dtype again just in case
                 print(f"Advertencia: Canales o tipos de dato diferentes después de conversión BGRA ({img1.shape}, {img1.dtype} vs {img2.shape}, {img2.dtype}). Intentando homogeneizar a BGR.")
                 try:
                     # Convertir a BGR si es escala de grises
                     if img1.ndim == 2:
                         img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
                         channels1 = 3
                         print("Imagen 1 escala de grises convertida a BGR.")
                     if img2.ndim == 2:
                         img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
                         channels2 = 3
                         print("Imagen 2 escala de grises convertida a BGR.")

                     # Verificar que después de la conversión tengan el mismo número de canales
                     if channels1 != channels2:
                          raise ValueError(f"Las imágenes tienen diferente número de canales ({channels1} vs {channels2}) después de intentar homogeneizar a BGR.")

                     # Asegurarse de que ambas sean uint8 (ya se hizo arriba, pero doble chequeo)
                     if img1.dtype != np.uint8: img1 = np.clip(img1, 0, 255).astype(np.uint8)
                     if img2.dtype != np.uint8: img2 = np.clip(img2, 0, 255).astype(np.uint8)

                 except Exception as e:
                     return jsonify({'error': f'Error al homogeneizar canales/tipo de dato para operación binaria: {str(e)}'}), 400

            # Final check on dimensions after resizing and channel handling
            if img1.shape != img2.shape:
                 # This case should ideally not be reached if resizing worked correctly, but as a safeguard
                 return jsonify({'error': f'Las imágenes no tienen las mismas dimensiones ni canales después de la preparación: Imagen 1 {img1.shape}, Imagen 2 {img2.shape}'}), 400


        # --- Aplicar operación lógica ---
        result = None # Initialize result
        try:
            if operation == 'AND':
                result = cv2.bitwise_and(img1, img2)
            elif operation == 'OR':
                result = cv2.bitwise_or(img1, img2)
            elif operation == 'XOR':
                result = cv2.bitwise_xor(img1, img2)
            elif operation == 'NOT':
                result = cv2.bitwise_not(img1)
            # No 'else' needed here for operation, as invalid operations were handled above

            if result is None:
                 # Should not happen if operation is valid and images are prepared
                 raise ValueError("La operación lógica no produjo un resultado válido.")

        except cv2.error as e:
             return jsonify({'error': f'Error de OpenCV al aplicar operación lógica {operation}: {str(e)}'}), 500
        except Exception as e:
            # Catch any other unexpected errors during the bitwise operation itself
            return jsonify({'error': f'Error inesperado al aplicar operación lógica {operation}: {str(e)}'}), 500


        # Guardar y devolver resultado
        # En el contexto de la API, no modificamos image_data['image1'] o ['image2']
        # Simplemente guardamos el resultado en 'processed_image' si es relevante
        # o lo devolvemos directamente. Aquí, lo devolvemos.
        # Si necesitas actualizar el processed_image en el backend, puedes añadir esa lógica.
        # Por ahora, la API simplemente devuelve el resultado.
        # image_data['processed_image'] = result # Opcional: guardar en el estado del backend si es necesario

        img_base64 = cv2_to_base64(result) # Convertir el resultado a base64
        if img_base64 is None:
             return jsonify({'error': 'Error al codificar la imagen resultante a base64'}), 500


        return jsonify({
            'success': True,
            'image': img_base64,
            'message': f'Operación lógica {operation} aplicada exitosamente'
        })

    except Exception as e:
        # Este catch final es para cualquier error que no haya sido manejado antes
        return jsonify({'error': f'Error inesperado en la ruta apply_logical: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0') 