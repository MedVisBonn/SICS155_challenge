# feature extraction pipeline.
# features should be extracted directly from a video file and saved as numpy arrays
# afterwards the features are passed to the model for training or inference

import cv2
import numpy as np
import os
import torch
from PIL import Image
import torchvision.transforms as transforms

def extract_frames(video_path, fps=15, use_gpu=True, target_size=(224, 224)):
    """
    Extract frames from a video at a specified frame rate and resize them.
    
    Args:
        video_path (str): Path to the video file
        fps (int): Target frames per second for extraction (default: 15)
        use_gpu (bool): Whether to use GPU acceleration if available (default: True)
        target_size (tuple): Target width and height for resizing (default: (224, 224))
        
    Returns:
        np.ndarray: Array of video frames with shape (num_frames, height, width, channels)
    """
    # Check if the file exists
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check GPU availability more safely
    cv2_cuda_available = False
    try:
        # Attempt to check CUDA availability safely
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Try a simple CUDA operation to verify it works
            test_mat = cv2.cuda_GpuMat((1, 1), cv2.CV_8UC3)
            test_mat.release()
            cv2_cuda_available = True
    except Exception as e:
        print(f"WARNING: CUDA error detected in OpenCV: {e}")
        cv2_cuda_available = False
    
    if use_gpu and not cv2_cuda_available:
        print("WARNING: GPU acceleration requested but CUDA is not available or working in OpenCV. Falling back to CPU.")
        use_gpu = False
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate frame extraction interval
    interval = int(round(original_fps / fps))
    
    # Initialize list to store frames
    frames = []
    
    # Read frames at the desired interval
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % interval == 0:
            if use_gpu:
                # Upload frame to GPU
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                
                # Convert from BGR to RGB on GPU
                gpu_frame_rgb = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2RGB)
                
                # Resize on GPU
                gpu_resized = cv2.cuda.resize(gpu_frame_rgb, target_size)
                
                # Download the result back to CPU
                frame_processed = gpu_resized.download()
                frames.append(frame_processed)
            else:
                # Convert from BGR to RGB on CPU
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize on CPU
                frame_resized = cv2.resize(frame_rgb, target_size)
                frames.append(frame_resized)
        
        frame_idx += 1
    
    # Release the video capture object
    cap.release()
    
    # Check if we extracted any frames
    if not frames:
        raise ValueError(f"No frames were extracted from {video_path}")
    
    # Convert list of frames to numpy array
    return np.array(frames)

def load_video(video_path, fps=15, use_gpu=True, target_size=(224, 224)):
    """
    Load video and convert to numpy array of frames.
    
    Args:
        video_path (str): Path to the video file
        fps (int): Target frames per second
        use_gpu (bool): Whether to use GPU acceleration if available
        target_size (tuple): Target width and height for resizing (default: (224, 224))
        
    Returns:
        np.ndarray: Array of video frames
    """
    return extract_frames(video_path, fps, use_gpu, target_size)

def resize_frames_gpu(frames, target_size=(224, 224)):
    """
    Resize frames using GPU acceleration.
    
    Args:
        frames (np.ndarray): Array of frames with shape (num_frames, height, width, channels)
        target_size (tuple): Target width and height (default: (224, 224))
        
    Returns:
        np.ndarray: Array of resized frames
    """
    # Check if CUDA is available and working
    cuda_available = False
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            # Test if CUDA actually works
            test_mat = cv2.cuda_GpuMat((1, 1), cv2.CV_8UC3)
            test_mat.release()
            cuda_available = True
    except Exception as e:
        print(f"WARNING: CUDA error in OpenCV: {e}")
        cuda_available = False
    
    if not cuda_available:
        # Fall back to CPU if CUDA is not available
        print("CUDA not available in OpenCV. Using CPU for frame resizing.")
        resized_frames = []
        for frame in frames:
            resized = cv2.resize(frame, target_size)
            resized_frames.append(resized)
        return np.array(resized_frames)
    
    resized_frames = []
    
    for frame in frames:
        # Upload frame to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Resize on GPU
        gpu_resized = cv2.cuda.resize(gpu_frame, target_size)
        
        # Download the result
        resized = gpu_resized.download()
        resized_frames.append(resized)
    
    return np.array(resized_frames)


def preprocess_image(image_array, target_size=(224, 224)):
    """
    Preprocess an image array for neural network inference.
    
    Args:
        image_array: NumPy array of shape (H, W, C) in RGB format
        target_size: Tuple of (height, width) to resize the image to
    
    Returns:
        torch.Tensor: Processed tensor ready for model inference
    """
    # ImageNet normalization values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Create transformation pipeline
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # Convert numpy array to PIL Image if needed
    if isinstance(image_array, np.ndarray):
        # Handle different input formats
        if image_array.ndim == 2:  # Grayscale
            image = Image.fromarray(image_array, mode='L').convert('RGB')
        elif image_array.shape[2] == 1:  # Grayscale with channel dim
            image = Image.fromarray(image_array[:,:,0], mode='L').convert('RGB')
        elif image_array.shape[2] == 3:  # RGB
            image = Image.fromarray(image_array.astype(np.uint8))
        elif image_array.shape[2] == 4:  # RGBA
            image = Image.fromarray(image_array.astype(np.uint8)[:,:,:3])
        else:
            raise ValueError(f"Unsupported image shape: {image_array.shape}")
    else:
        raise TypeError("Input must be a numpy array")
    
    # Apply transformations and add batch dimension
    tensor = preprocess(image).unsqueeze(0)
    
    return tensor


def preprocess_frames(frames, target_size=(224, 224), batch=True):
    """
    Preprocess multiple frames for model inference.
    
    Args:
        frames: NumPy array of shape (num_frames, H, W, C) in RGB format
        target_size: Tuple of (height, width) for resizing
        batch: Whether to return a batched tensor (default: True)
    
    Returns:
        torch.Tensor: Processed tensor ready for model inference,
                      shape (batch_size, C, H, W) if batch=True
                      or list of (1, C, H, W) tensors if batch=False
    """
    if batch:
        # Process all frames as a batch
        tensors = []
        for frame in frames:
            # Process without adding batch dimension
            tensor = preprocess_image(frame, target_size=target_size)
            tensors.append(tensor)
        # Stack all tensors along batch dimension
        return torch.cat(tensors, dim=0)
    else:
        # Return a list of individual tensors
        return [preprocess_image(frame, target_size=target_size) for frame in frames]


def extract_and_preprocess_video(video_path, fps=15, use_gpu=True, target_size=(224, 224), batch=True):
    """
    Extract frames from a video and preprocess them for model inference.
    
    Args:
        video_path (str): Path to the video file
        fps (int): Target frames per second
        use_gpu (bool): Whether to use GPU for frame extraction
        target_size (tuple): Target size for resizing
        batch (bool): Whether to return a batched tensor
        
    Returns:
        torch.Tensor: Processed tensor ready for model inference
    """
    # Extract frames
    frames = extract_frames(video_path, fps, use_gpu, target_size)
    
    # Preprocess frames
    return preprocess_frames(frames, target_size, batch)