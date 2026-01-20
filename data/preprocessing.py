"""Preprocessing utilities for deepfake detection."""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    print("Warning: facenet-pytorch not available. Face detection will use fallback method.")


class FaceExtractor:
    """Face extraction using MTCNN or fallback to Haar Cascade."""
    
    def __init__(
        self,
        device: str = 'cpu',
        image_size: int = 224,
        margin: int = 20,
        post_process: bool = True
    ):
        """
        Initialize face extractor.
        
        Args:
            device: Device to run face detection on ('cpu' or 'cuda')
            image_size: Target size for extracted faces
            margin: Margin around detected face
            post_process: Whether to post-process the detected faces
        """
        self.device = device
        self.image_size = image_size
        self.margin = margin
        self.post_process = post_process
        
        if MTCNN_AVAILABLE:
            self.detector = MTCNN(
                image_size=image_size,
                margin=margin,
                device=device,
                post_process=post_process,
                keep_all=False,  # Only keep the most confident face
                select_largest=True
            )
        else:
            # Fallback to OpenCV Haar Cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
    
    def extract_face(
        self,
        image: np.ndarray,
        return_bbox: bool = False
    ) -> Optional[np.ndarray]:
        """
        Extract face from image.
        
        Args:
            image: Input image as numpy array (H, W, C)
            return_bbox: Whether to return bounding box coordinates
            
        Returns:
            Extracted face as numpy array or None if no face detected
            If return_bbox is True, returns (face, bbox)
        """
        if MTCNN_AVAILABLE:
            return self._extract_with_mtcnn(image, return_bbox)
        else:
            return self._extract_with_cascade(image, return_bbox)
    
    def _extract_with_mtcnn(
        self,
        image: np.ndarray,
        return_bbox: bool = False
    ) -> Optional[np.ndarray]:
        """Extract face using MTCNN."""
        # MTCNN expects PIL Image
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Detect face
        face_tensor, prob = self.detector(image_pil, return_prob=True)
        
        if face_tensor is None:
            return None if not return_bbox else (None, None)
        
        # Convert tensor to numpy
        face = face_tensor.permute(1, 2, 0).numpy()
        face = (face * 255).astype(np.uint8)
        
        if return_bbox:
            # MTCNN doesn't directly return bbox in this mode
            # Return None for bbox
            return face, None
        
        return face
    
    def _extract_with_cascade(
        self,
        image: np.ndarray,
        return_bbox: bool = False
    ) -> Optional[np.ndarray]:
        """Extract face using Haar Cascade."""
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Detect faces
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            return None if not return_bbox else (None, None)
        
        # Take the largest face
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        
        # Add margin
        x = max(0, x - self.margin)
        y = max(0, y - self.margin)
        w = min(image.shape[1] - x, w + 2 * self.margin)
        h = min(image.shape[0] - y, h + 2 * self.margin)
        
        # Crop face
        face = image[y:y+h, x:x+w]
        
        # Resize to target size
        face = cv2.resize(face, (self.image_size, self.image_size))
        
        if return_bbox:
            return face, (x, y, w, h)
        
        return face


def extract_frames(
    video_path: str,
    num_frames: int = 10,
    sample_method: str = 'uniform'
) -> List[np.ndarray]:
    """
    Extract frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        sample_method: Method to sample frames ('uniform', 'random')
        
    Returns:
        List of frames as numpy arrays
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print(f"Warning: Could not read video {video_path}")
        cap.release()
        return []
    
    # Determine frame indices to extract
    if sample_method == 'uniform':
        # Uniformly sample frames
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    elif sample_method == 'random':
        # Randomly sample frames
        frame_indices = np.random.choice(total_frames, size=min(num_frames, total_frames), replace=False)
        frame_indices.sort()
    else:
        raise ValueError(f"Unknown sample method: {sample_method}")
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames


def preprocess_image(
    image_path: str,
    image_size: int = 224,
    normalize: bool = True
) -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image_path: Path to image file
        image_size: Target image size
        normalize: Whether to normalize with ImageNet statistics
        
    Returns:
        Preprocessed image as numpy array
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize
    image = image.resize((image_size, image_size))
    
    # Convert to numpy array
    image = np.array(image).astype(np.float32)
    
    # Normalize
    if normalize:
        image = image / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
    
    return image


def validate_dataset(
    dataset_dir: str,
    required_splits: List[str] = ['train', 'val', 'test']
) -> bool:
    """
    Validate dataset structure.
    
    Args:
        dataset_dir: Root directory of dataset
        required_splits: Required dataset splits
        
    Returns:
        True if dataset is valid, False otherwise
    """
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_dir} does not exist")
        return False
    
    for split in required_splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Warning: Split directory {split} not found")
            return False
    
    return True


def save_extracted_faces(
    video_dir: str,
    output_dir: str,
    face_extractor: FaceExtractor,
    num_frames_per_video: int = 10
):
    """
    Extract and save faces from videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save extracted faces
        face_extractor: FaceExtractor instance
        num_frames_per_video: Number of frames to extract per video
    """
    video_path = Path(video_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_path.glob(f'*{ext}'))
    
    print(f"Found {len(video_files)} videos in {video_dir}")
    
    for video_file in video_files:
        # Extract frames
        frames = extract_frames(str(video_file), num_frames_per_video)
        
        # Extract faces from frames
        for i, frame in enumerate(frames):
            face = face_extractor.extract_face(frame)
            if face is not None:
                # Save face
                face_filename = f"{video_file.stem}_frame{i:03d}.jpg"
                face_path = output_path / face_filename
                Image.fromarray(face).save(face_path)
    
    print(f"Extracted faces saved to {output_dir}")
