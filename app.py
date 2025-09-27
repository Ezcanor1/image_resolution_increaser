import os
import base64
import io
import time
import urllib.request
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify
import cv2
from pathlib import Path
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Configure folders
UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'models'
GAN_MODELS_FOLDER = os.path.join(MODELS_FOLDER, 'gan')
DL_MODELS_FOLDER = os.path.join(MODELS_FOLDER, 'dl')

# Create necessary directories
for folder in [UPLOAD_FOLDER, MODELS_FOLDER, GAN_MODELS_FOLDER, DL_MODELS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to track available models
AI_MODELS = {
    'sr_models': {},
    'gan_models': {},
    'dl_models': {}
}

# Check for GPU availability
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Function to download OpenCV DNN models
def download_opencv_models():
    models = {
        'EDSR_x4.pb': "https://github.com/Saafke/EDSR_Tensorflow/raw/master/models/EDSR_x4.pb",
        'FSRCNN_x4.pb': "https://github.com/fannymonori/TF-FSRCNN/raw/master/FSRCNN_x4.pb",
        'LapSRN_x4.pb': "https://github.com/fannymonori/TF-LapSRN/raw/master/LapSRN_x4.pb",
        'ESPCN_x4.pb': "https://github.com/fannymonori/TF-ESPCN/raw/master/ESPCN_x4.pb"
    }
    
    downloaded_models = []
    for model_name, url in models.items():
        model_path = os.path.join(DL_MODELS_FOLDER, model_name)
        if not os.path.exists(model_path):
            try:
                print(f"Downloading {model_name}...")
                urllib.request.urlretrieve(url, model_path)
                downloaded_models.append(model_name)
            except Exception as e:
                print(f"Failed to download {model_name}: {e}")
        else:
            downloaded_models.append(model_name)
    
    return downloaded_models

# RRDB (Residual in Residual Dense Block) Network for ESRGAN
class RRDB(nn.Module):
    def __init__(self, channels=64, growth_channels=32, res_scale=0.2):
        super(RRDB, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.ModuleList([self._make_dense_block(channels, growth_channels) for _ in range(3)])
        
    def _make_dense_block(self, channels, growth_channels):
        layers = []
        for i in range(5):
            in_channels = channels + i * growth_channels
            layers.append(nn.Sequential(
                nn.Conv2d(in_channels, growth_channels, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True)
            ))
        return nn.ModuleList(layers)
        
    def forward(self, x):
        res = x
        for dense_block in self.dense_blocks:
            dense_out = x
            for layer in dense_block:
                out = layer(dense_out)
                dense_out = torch.cat([dense_out, out], dim=1)
            x = x + self.res_scale * dense_out
        return res + self.res_scale * x

# Simplified ESRGAN for image super-resolution
class ESRGAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, nf=64, gc=32, scale=4):
        super(ESRGAN, self).__init__()
        self.conv_first = nn.Conv2d(in_channels, nf, 3, 1, 1)
        
        # Use 8 RRDB blocks (simplified from 23 in original)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(8)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # Upsampling layers
        if scale == 4:
            self.upsampling = nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2),
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
        else:  # scale == 2
            self.upsampling = nn.Sequential(
                nn.Conv2d(nf, nf * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            
        self.conv_last = nn.Conv2d(nf, out_channels, 3, 1, 1)
        
    def forward(self, x):
        feat_first = self.conv_first(x)
        body_out = self.conv_body(self.body(feat_first))
        feat = feat_first + body_out
        out = self.upsampling(feat)
        out = self.conv_last(out)
        return out

class ImageEnhancementAttentionNet(nn.Module):
    """A custom network with attention mechanism for enhancing image details"""
    def __init__(self, channels=3):
        super(ImageEnhancementAttentionNet, self).__init__()
        # Initial features extraction
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Attention module
        self.attention_conv = nn.Conv2d(64, 64, kernel_size=1, padding=0)
        self.attention_sigmoid = nn.Sigmoid()
        
        # Feature processing
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial feature extraction
        feat1 = F.relu(self.conv1(x))
        feat2 = F.relu(self.conv2(feat1))
        
        # Attention mechanism
        attention_map = self.attention_sigmoid(self.attention_conv(feat2))
        attended_feat = feat2 * attention_map
        
        # Final processing
        feat3 = F.relu(self.conv3(attended_feat))
        out = self.conv4(feat3)
        
        # Residual connection
        return x + out

# Create simplified models for demo purposes
def create_sample_models():
    # Simple GAN-based enhancer model
    esrgan_model = ESRGAN(scale=2)  # Simplified ESRGAN
    esrgan_path = os.path.join(GAN_MODELS_FOLDER, 'esrgan_sample.pth')
    
    # Save the model if it doesn't exist
    if not os.path.exists(esrgan_path):
        torch.save(esrgan_model.state_dict(), esrgan_path)
    
    # Enhancement network
    enhancer = ImageEnhancementAttentionNet()
    enhancer_path = os.path.join(DL_MODELS_FOLDER, 'enhancer_sample.pth')
    
    # Save the model if it doesn't exist
    if not os.path.exists(enhancer_path):
        torch.save(enhancer.state_dict(), enhancer_path)
    
    return {'esrgan': esrgan_path, 'enhancer': enhancer_path}

# Initialize and load all available models
def initialize_models():
    # 1. Load OpenCV DNN super-resolution models
    try:
        downloaded_models = download_opencv_models()
        for model_name in downloaded_models:
            model_path = os.path.join(DL_MODELS_FOLDER, model_name)
            model_type = model_name.split('_')[0].lower()
            scale = int(model_name.split('_x')[1].split('.')[0])
            
            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(model_path)
                sr.setModel(model_type, scale)
                AI_MODELS['sr_models'][model_type] = sr
                print(f"Loaded OpenCV SR model: {model_name}")
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
    except Exception as e:
        print(f"Could not initialize OpenCV super-resolution models: {e}")

    # 2. Create and load PyTorch models
    try:
        sample_models = create_sample_models()
        
        # Load ESRGAN
        esrgan_model = ESRGAN(scale=2)
        esrgan_model.load_state_dict(torch.load(sample_models['esrgan'], map_location=DEVICE))
        esrgan_model.to(DEVICE)
        esrgan_model.eval()
        AI_MODELS['gan_models']['esrgan'] = esrgan_model
        print("Loaded ESRGAN sample model")
        
        # Load enhancer
        enhancer = ImageEnhancementAttentionNet()
        enhancer.load_state_dict(torch.load(sample_models['enhancer'], map_location=DEVICE))
        enhancer.to(DEVICE)
        enhancer.eval()
        AI_MODELS['dl_models']['enhancer'] = enhancer
        print("Loaded Enhancer sample model")
    except Exception as e:
        print(f"Could not initialize PyTorch models: {e}")

# Try to initialize models at startup
initialize_models()

# Determine which models are available
HAVE_SR_MODELS = len(AI_MODELS['sr_models']) > 0
HAVE_GAN_MODELS = len(AI_MODELS['gan_models']) > 0
HAVE_DL_MODELS = len(AI_MODELS['dl_models']) > 0

print(f"Available models - SR: {HAVE_SR_MODELS}, GAN: {HAVE_GAN_MODELS}, DL: {HAVE_DL_MODELS}")

def apply_deep_learning_sr(img, model_name='edsr'):
    """Apply deep learning super-resolution using OpenCV DNN models"""
    if model_name in AI_MODELS['sr_models']:
        return AI_MODELS['sr_models'][model_name].upsample(img)
    else:
        # If requested model not available, use the first available one
        for name, model in AI_MODELS['sr_models'].items():
            return model.upsample(img)
    return img  # Return original if no models available

def apply_gan_upscaling(img, model_name='esrgan'):
    """Apply GAN-based upscaling to the image"""
    if model_name not in AI_MODELS['gan_models']:
        return img  # Return original if model not available
    
    # Convert BGR to RGB and normalize to [0, 1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        model = AI_MODELS['gan_models'][model_name]
        output = model(img_tensor)
        
    # Convert back to numpy array [0, 255] and RGB to BGR
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return output

def apply_detail_enhancement(img, model_name='enhancer'):
    """Apply neural detail enhancement to the image"""
    if model_name not in AI_MODELS['dl_models']:
        return img  # Return original if model not available
    
    # Convert BGR to RGB and normalize to [0, 1]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(DEVICE)
    
    with torch.no_grad():
        model = AI_MODELS['dl_models'][model_name]
        output = model(img_tensor)
        
    # Convert back to numpy array [0, 255] and RGB to BGR
    output = output.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0
    output = cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    return output

def apply_lab_enhancement(image):
    """Enhance image using LAB color space for better color processing"""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge enhanced L with original A and B
    lab_enhanced = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    return cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

def apply_wavelet_denoising(image):
    """Advanced wavelet-based denoising preserving edges"""
    # Convert to YCrCb color space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    
    # Apply wavelet denoising to Y channel only (simulated)
    y_denoised = cv2.fastNlMeansDenoising(y, None, 10, 7, 21)
    
    # Merge channels
    ycrcb_denoised = cv2.merge([y_denoised, cr, cb])
    
    # Convert back to BGR
    return cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2BGR)

def apply_adaptive_sharpening(image):
    """Apply adaptive sharpening based on local gradient"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = cv2.magnitude(sobelx, sobely)
    
    # Normalize to [0, 1]
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 1, cv2.NORM_MINMAX)
    
    # Threshold for edge detection
    edges = (gradient_mag > 0.2).astype(np.float32)
    
    # Apply unsharp mask with varying strength
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    # Convert to float32 for weighted blending
    image_f32 = image.astype(np.float32)
    sharpened_f32 = sharpened.astype(np.float32)
    
    # Expand edges to 3-channel for multiplication
    edges_3ch = np.stack([edges, edges, edges], axis=2)
    
    # Apply adaptive sharpening - more at edges, less in flat areas
    result = image_f32 * (1 - edges_3ch) + sharpened_f32 * edges_3ch
    
    return np.clip(result, 0, 255).astype(np.uint8)

def dnn_refinement(image):
    """Refines image details using a combination of techniques inspired by deep learning approaches"""
    # Extract and enhance edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated_edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edge_mask = dilated_edges.astype(np.float32) / 255.0
    
    # Create a detail layer using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    laplacian = np.abs(laplacian)
    laplacian = cv2.normalize(laplacian, None, 0, 1, cv2.NORM_MINMAX)
    
    # Combine edge and detail information
    detail_mask = cv2.addWeighted(edge_mask, 0.7, laplacian, 0.3, 0)
    detail_mask = cv2.GaussianBlur(detail_mask, (0, 0), 1)
    
    # Apply subtle sharpening at detail areas
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    
    # Convert to float for blending
    image_f32 = image.astype(np.float32)
    sharpened_f32 = sharpened.astype(np.float32)
    
    # Apply adaptive enhancement
    detail_mask_3ch = np.stack([detail_mask, detail_mask, detail_mask], axis=2)
    blending_factor = 0.4  # Control strength of enhancement
    result = image_f32 * (1.0 - blending_factor * detail_mask_3ch) + \
             sharpened_f32 * (blending_factor * detail_mask_3ch)
    
    return np.clip(result, 0, 255).astype(np.uint8)

def multi_stage_upscaling(image, target_scale, method='ai_ensemble', 
                         enhance_details=True, reduce_noise=False, sharpen=True, 
                         preserve_colors=True, contrast=True):
    """
    Multi-stage upscaling using ensemble of AI models and advanced image processing
    
    Args:
        image: Input image
        target_scale: Target scale factor
        method: Upscaling method
        enhance_details: Whether to enhance details
        reduce_noise: Whether to apply noise reduction
        sharpen: Whether to apply sharpening
        preserve_colors: Whether to preserve original colors
        contrast: Whether to enhance contrast
    
    Returns:
        Upscaled image
    """
    # Store original color information if preserving colors
    if preserve_colors:
        original_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(original_lab)
        original_color_channels = (a, b)
    
    # Step 1: Optional Pre-processing
    if reduce_noise:
        # Use advanced wavelet-based denoising
        image = apply_wavelet_denoising(image)
    
    # Store original dimensions
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = int(orig_h * target_scale), int(orig_w * target_scale)
    
    # Step 2: Initial Upscaling based on method
    if method == 'ai_ensemble' and (HAVE_SR_MODELS or HAVE_GAN_MODELS):
        # Use an ensemble of available AI upscalers
        upscaled_imgs = []
        weights = []
        
        # Apply OpenCV DNN SR models if available
        if HAVE_SR_MODELS:
            # Preference order for models
            model_preference = ['edsr', 'lapsrn', 'fsrcnn', 'espcn']
            for model_name in model_preference:
                if model_name in AI_MODELS['sr_models']:
                    try:
                        # All OpenCV models are 4x
                        sr_result = apply_deep_learning_sr(image, model_name)
                        
                        # Resize if needed to target size
                        if sr_result.shape[:2] != (target_h, target_w):
                            sr_result = cv2.resize(sr_result, (target_w, target_h), 
                                                  interpolation=cv2.INTER_LANCZOS4)
                            
                        upscaled_imgs.append(sr_result)
                        weights.append(0.7)  # EDSR gets higher weight
                        break  # Use only the best available model
                    except Exception as e:
                        print(f"Error using {model_name}: {e}")
        
        # Apply GAN model if available
        if HAVE_GAN_MODELS and 'esrgan' in AI_MODELS['gan_models']:
            try:
                # GAN models often handle only 2x or 4x
                # Apply iteratively for higher scales if needed
                gan_result = image.copy()
                current_scale = 1.0
                
                # Apply incremental 2x upscales until we reach target scale
                while current_scale * 2 <= target_scale:
                    gan_result = apply_gan_upscaling(gan_result, 'esrgan')
                    current_scale *= 2
                
                # Final resize to target if needed
                if gan_result.shape[:2] != (target_h, target_w):
                    gan_result = cv2.resize(gan_result, (target_w, target_h), 
                                          interpolation=cv2.INTER_LANCZOS4)
                
                upscaled_imgs.append(gan_result)
                weights.append(0.8)  # GAN model gets highest weight
            except Exception as e:
                print(f"Error using GAN model: {e}")
        
        # If no AI models succeeded, use Lanczos
        if not upscaled_imgs:
            upscaled = cv2.resize(image, (target_w, target_h), 
                                 interpolation=cv2.INTER_LANCZOS4)
        else:
            # Normalize weights
            weights = [w/sum(weights) for w in weights]
            
            # Weighted blend of all upscaled results
            upscaled = np.zeros_like(upscaled_imgs[0], dtype=np.float32)
            for img, weight in zip(upscaled_imgs, weights):
                upscaled += img.astype(np.float32) * weight
                
            upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
    else:
        # Fallback to high-quality Lanczos upscaling
        upscaled = cv2.resize(image, (target_w, target_h), 
                             interpolation=cv2.INTER_LANCZOS4)
    
    # Step 3: Post-processing pipeline
    
    # 3.1: Color preservation
    if preserve_colors and upscaled.shape[0] > 0 and upscaled.shape[1] > 0:
        try:
            # Convert upscaled image to LAB
            upscaled_lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
            # Extract luminance channel from upscaled image
            upscaled_l, _, _ = cv2.split(upscaled_lab)
            
            # Resize original color channels to match new dimensions
            resized_a = cv2.resize(original_color_channels[0], (upscaled.shape[1], upscaled.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
            resized_b = cv2.resize(original_color_channels[1], (upscaled.shape[1], upscaled.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
            
            # Create new LAB image with upscaled luminance but original colors
            merged_lab = cv2.merge([upscaled_l, resized_a, resized_b])
            
            # Convert back to BGR
            upscaled = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        except Exception as e:
            print(f"Color preservation failed: {e}")
    
    # 3.2: Apply detail enhancement with neural network if available
    if enhance_details and HAVE_DL_MODELS and 'enhancer' in AI_MODELS['dl_models']:
        try:
            upscaled = apply_detail_enhancement(upscaled, 'enhancer')
        except Exception as e:
            print(f"Neural detail enhancement failed: {e}")
            # Fallback to traditional detail enhancement
            if enhance_details:
                upscaled = dnn_refinement(upscaled)
    elif enhance_details:
        # Use traditional detail enhancement methods
        upscaled = dnn_refinement(upscaled)
    
    # 3.3: Apply contrast enhancement
    if contrast:
        try:
            upscaled = apply_lab_enhancement(upscaled)
        except Exception as e:
            print(f"Contrast enhancement failed: {e}")
    
    # 3.4: Apply adaptive sharpening
    if sharpen:
        try:
            upscaled = apply_adaptive_sharpening(upscaled)
        except Exception as e:
            print(f"Adaptive sharpening failed: {e}")
            # Fallback to basic sharpening
            kernel = np.array([[-1, -1, -1], 
                              [-1,  9, -1], 
                              [-1, -1, -1]])
            upscaled = cv2.filter2D(upscaled, -1, kernel)
    
    # Ensure valid pixel values
    upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)
    
    return upscaled

# HTML templates
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upscale', methods=['POST'])
def upscale():
    # Check if an image was sent
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Get parameters
    scale_factor = float(request.form.get('scale', 2.0))
    method = request.form.get('method', 'ai_ensemble')
    enhance_details = request.form.get('enhance_details', 'true').lower() == 'true'
    reduce_noise = request.form.get('reduce_noise', 'false').lower() == 'true'  # Default to false
    sharpen = request.form.get('sharpen', 'true').lower() == 'true'
    preserve_colors = request.form.get('preserve_colors', 'true').lower() == 'true'
    contrast = request.form.get('contrast', 'true').lower() == 'true'
    
    # Read image
    try:
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Invalid image format or corrupted image'}), 400
        
        # Get original dimensions for display
        original_height, original_width = img.shape[:2]
        
        # For very large images, resize down before processing to prevent memory issues
        max_dimension = 1500  # Maximum dimension for processing
        resize_factor = 1.0
        if max(original_height, original_width) > max_dimension:
            resize_factor = max_dimension / max(original_height, original_width)
            img = cv2.resize(img, (int(original_width * resize_factor), int(original_height * resize_factor)), 
                           interpolation=cv2.INTER_AREA)
            # Adjust scale factor to compensate for initial resize
            scale_factor = scale_factor / resize_factor
        
        # Track processing time
        start_time = time.time()
        
        # Process the image with the multi-stage approach
        upscaled_img = multi_stage_upscaling(
            img, 
            target_scale=scale_factor, 
            method=method,
            enhance_details=enhance_details,
            reduce_noise=reduce_noise,
            sharpen=sharpen,
            preserve_colors=preserve_colors,
            contrast=contrast
        )
        
        new_height, new_width = upscaled_img.shape[:2]
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Convert the image to base64 for display
        _, buffer = cv2.imencode('.png', upscaled_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'image': f'data:image/png;base64,{img_str}',
            'original_dimensions': f'{original_width}x{original_height}',
            'new_dimensions': f'{new_width}x{new_height}',
            'processing_time': f'{processing_time:.2f} seconds',
            'upscale_method': method,
            'models_used': {
                'sr_models': list(AI_MODELS['sr_models'].keys()),
                'gan_models': list(AI_MODELS['gan_models'].keys()),
                'dl_models': list(AI_MODELS['dl_models'].keys())
            }
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/download', methods=['POST'])
def download():
    if 'image_data' not in request.form:
        return jsonify({'error': 'No image data'}), 400
    
    try:
        # Get the base64 encoded image
        image_data = request.form['image_data'].split(',')[1]  # Remove the prefix
        image_bytes = base64.b64decode(image_data)
        
        # Create a BytesIO object for sending the file
        img_io = io.BytesIO(image_bytes)
        img_io.seek(0)
        
        return send_file(
            img_io,
            mimetype='image/png',
            as_attachment=True,
            download_name='enhanced_image.png'
        )
    except Exception as e:
        return jsonify({'error': f'Error creating download: {str(e)}'}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Return available models and capabilities"""
    return jsonify({
        'sr_models': list(AI_MODELS['sr_models'].keys()),
        'gan_models': list(AI_MODELS['gan_models'].keys()),
        'dl_models': list(AI_MODELS['dl_models'].keys()),
        'capabilities': {
            'super_resolution': HAVE_SR_MODELS,
            'gan_enhancement': HAVE_GAN_MODELS,
            'detail_enhancement': HAVE_DL_MODELS
        }
    })

# Create an HTML template for the frontend
@app.route('/templates/index.html')
def serve_index_template():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI Image Enhancer</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .image-container {
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }
            .image-box {
                display: flex;
                flex-direction: column;
                align-items: center;
                flex: 1;
                min-width: 300px;
                max-width: 500px;
            }
            .image-box img {
                max-width: 100%;
                max-height: 500px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }
            .options {
                margin-bottom: 20px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .options h3 {
                margin-top: 0;
                color: #2c3e50;
            }
            .form-group {
                margin-bottom: 15px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: 500;
            }
            select, input[type="range"], input[type="file"], input[type="checkbox"] {
                width: 100%;
                padding: 8px;
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: white;
            }
            input[type="checkbox"] {
                width: auto;
            }
            button {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                transition: background-color 0.3s;
            }
            button:hover {
                background-color: #2980b9;
            }
            #downloadBtn {
                background-color: #27ae60;
            }
            #downloadBtn:hover {
                background-color: #219955;
            }
            .info {
                font-size: 14px;
                color: #666;
                margin-top: 5px;
            }
            .checkbox-group {
                display: flex;
                gap: 15px;
                flex-wrap: wrap;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .results {
                display: flex;
                justify-content: space-between;
                margin-top: 10px;
                font-size: 14px;
                color: #666;
            }
            .loading {
                text-align: center;
                display: none;
            }
            .loading-spinner {
                border: 5px solid #f3f3f3;
                border-top: 5px solid #3498db;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 20px auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            #methodInfo {
                font-style: italic;
                display: block;
                margin-top: 5px;
                color: #666;
            }
            .model-info {
                font-size: 14px;
                color: #666;
                margin-top: 20px;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
            .slider-value {
                display: inline-block;
                min-width: 30px;
                text-align: right;
            }
        </style>
    </head>
    <body>
        <h1>Ultimate AI Image Enhancer</h1>
        <div class="container">
            <div class="options">
                <h3>Enhancement Options</h3>
                <form id="uploadForm">
                    <div class="form-group">
                        <label for="imageFile">Select Image:</label>
                        <input type="file" id="imageFile" name="image" accept="image/*">
                    </div>
                    <div class="form-group">
                        <label for="scaleSlider">Scale Factor: <span id="scaleValue" class="slider-value">2.0</span>x</label>
                        <input type="range" id="scaleSlider" name="scale" min="1.0" max="4.0" step="0.1" value="2.0">
                    </div>
                    <div class="form-group">
                        <label for="method">Enhancement Method:</label>
                        <select id="method" name="method">
                            <option value="ai_ensemble">AI Ensemble (Best Quality)</option>
                            <option value="traditional">Traditional (Faster)</option>
                        </select>
                        <span id="methodInfo">AI Ensemble combines multiple neural networks for maximum quality</span>
                    </div>
                    <div class="form-group">
                        <label>Enhancement Options:</label>
                        <div class="checkbox-group">
                            <div class="checkbox-item">
                                <input type="checkbox" id="enhanceDetails" name="enhance_details" checked>
                                <label for="enhanceDetails">Enhance Details</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="reduceNoise" name="reduce_noise">
                                <label for="reduceNoise">Reduce Noise</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="sharpen" name="sharpen" checked>
                                <label for="sharpen">Sharpen</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="preserveColors" name="preserve_colors" checked>
                                <label for="preserveColors">Preserve Colors</label>
                            </div>
                            <div class="checkbox-item">
                                <input type="checkbox" id="contrast" name="contrast" checked>
                                <label for="contrast">Enhance Contrast</label>
                            </div>
                        </div>
                    </div>
                    <button type="submit" id="enhanceBtn">Enhance Image</button>
                </form>
            </div>
            
            <div class="loading">
                <div class="loading-spinner"></div>
                <p>Processing image with AI models... Please wait</p>
            </div>
            
            <div class="model-info" id="modelInfo">
                Loading AI model information...
            </div>
            
            <div class="image-container">
                <div class="image-box">
                    <h3>Original Image</h3>
                    <img id="originalImage" alt="Original image will appear here">
                    <div class="info" id="originalInfo"></div>
                </div>
                <div class="image-box">
                    <h3>Enhanced Image</h3>
                    <img id="enhancedImage" alt="Enhanced image will appear here">
                    <div class="info" id="enhancedInfo"></div>
                    <div class="results" id="results"></div>
                    <button id="downloadBtn" style="display: none;">Download Enhanced Image</button>
                </div>
            </div>
        </div>

        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const uploadForm = document.getElementById('uploadForm');
                const enhanceBtn = document.getElementById('enhanceBtn');
                const downloadBtn = document.getElementById('downloadBtn');
                const originalImage = document.getElementById('originalImage');
                const enhancedImage = document.getElementById('enhancedImage');
                const originalInfo = document.getElementById('originalInfo');
                const enhancedInfo = document.getElementById('enhancedInfo');
                const results = document.getElementById('results');
                const loading = document.querySelector('.loading');
                const scaleSlider = document.getElementById('scaleSlider');
                const scaleValue = document.getElementById('scaleValue');
                const methodSelect = document.getElementById('method');
                const methodInfo = document.getElementById('methodInfo');
                const modelInfo = document.getElementById('modelInfo');
                
                // Update scale value display
                scaleSlider.addEventListener('input', function() {
                    scaleValue.textContent = parseFloat(this.value).toFixed(1);
                });
                
                // Update method info when selection changes
                methodSelect.addEventListener('change', function() {
                    if (this.value === 'ai_ensemble') {
                        methodInfo.textContent = 'AI Ensemble combines multiple neural networks for maximum quality';
                    } else {
                        methodInfo.textContent = 'Traditional method uses advanced algorithms without neural networks';
                    }
                });
                
                // Load model information
                fetch('/api/models')
                    .then(response => response.json())
                    .then(data => {
                        let infoText = 'Available AI models: ';
                        
                        if (data.capabilities.super_resolution) {
                            infoText += `<br>- Super Resolution: ${data.sr_models.join(', ')}`;
                        }
                        
                        if (data.capabilities.gan_enhancement) {
                            infoText += `<br>- GAN Enhancement: ${data.gan_models.join(', ')}`;
                        }
                        
                        if (data.capabilities.detail_enhancement) {
                            infoText += `<br>- Detail Enhancement: ${data.dl_models.join(', ')}`;
                        }
                        
                        if (!data.capabilities.super_resolution && 
                            !data.capabilities.gan_enhancement && 
                            !data.capabilities.detail_enhancement) {
                            infoText = 'No AI models available. Using traditional enhancement methods.';
                        }
                        
                        modelInfo.innerHTML = infoText;
                    })
                    .catch(error => {
                        modelInfo.textContent = 'Could not load AI model information.';
                        console.error('Error loading model information:', error);
                    });
                
                // Handle form submission
                uploadForm.addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const fileInput = document.getElementById('imageFile');
                    if (!fileInput.files[0]) {
                        alert('Please select an image first.');
                        return;
                    }
                    
                    const formData = new FormData(uploadForm);
                    
                    // Add checkbox values
                    formData.set('enhance_details', document.getElementById('enhanceDetails').checked);
                    formData.set('reduce_noise', document.getElementById('reduceNoise').checked);
                    formData.set('sharpen', document.getElementById('sharpen').checked);
                    formData.set('preserve_colors', document.getElementById('preserveColors').checked);
                    formData.set('contrast', document.getElementById('contrast').checked);
                    
                    // Show loading animation
                    enhanceBtn.disabled = true;
                    loading.style.display = 'block';
                    downloadBtn.style.display = 'none';
                    results.textContent = '';
                    
                    // Display original image
                    const file = fileInput.files[0];
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        originalImage.src = e.target.result;
                        originalImage.onload = function() {
                            originalInfo.textContent = `Dimensions: ${this.naturalWidth} × ${this.naturalHeight}`;
                        };
                    };
                    reader.readAsDataURL(file);
                    
                    // Send request to server
                    fetch('/upscale', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => {
                        if (!response.ok) {
                            return response.json().then(err => { throw new Error(err.error || 'Unknown error occurred'); });
                        }
                        return response.json();
                    })
                    .then(data => {
                        enhancedImage.src = data.image;
                        enhancedImage.onload = function() {
                            enhancedInfo.textContent = `Dimensions: ${data.new_dimensions}`;
                            results.textContent = `Processing time: ${data.processing_time}`;
                            
                            // Show download button
                            downloadBtn.style.display = 'inline-block';
                            downloadBtn.onclick = function() {
                                downloadEnhancedImage(data.image);
                            };
                            
                            // Additional info about models used
                            if (data.models_used) {
                                let modelsText = '<br><small>Models used: ';
                                if (data.models_used.sr_models.length) {
                                    modelsText += `SR: ${data.models_used.sr_models.join(', ')} `;
                                }
                                if (data.models_used.gan_models.length) {
                                    modelsText += `GAN: ${data.models_used.gan_models.join(', ')} `;
                                }
                                if (data.models_used.dl_models.length) {
                                    modelsText += `DL: ${data.models_used.dl_models.join(', ')}`;
                                }
                                modelsText += '</small>';
                                enhancedInfo.innerHTML += modelsText;
                            }
                        };
                    })
                    .catch(error => {
                        alert('Error: ' + error.message);
                        console.error('Error:', error);
                    })
                    .finally(() => {
                        // Hide loading animation
                        enhanceBtn.disabled = false;
                        loading.style.display = 'none';
                    });
                });
                
                function downloadEnhancedImage(imageData) {
                    fetch('/download', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                        },
                        body: `image_data=${encodeURIComponent(imageData)}`
                    })
                    .then(response => {
                        if (!response.ok) {
                            throw new Error('Download failed');
                        }
                        return response.blob();
                    })
                    .then(blob => {
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.style.display = 'none';
                        a.href = url;
                        a.download = 'enhanced_image.png';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    })
                    .catch(error => {
                        alert('Error downloading: ' + error.message);
                    });
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == '__main__':
    app.run(debug=True)