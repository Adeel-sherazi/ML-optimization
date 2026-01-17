"""
Real-time video inference pipeline with optimization.

Production-ready video processing with batching and streaming.
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
from collections import deque
import threading


class VideoPipeline:
    """Optimized video inference pipeline."""
    
    def __init__(
        self,
        model: nn.Module,
        input_size: Tuple[int, int] = (224, 224),
        batch_size: int = 4,
        use_cuda: bool = True
    ):
        """
        Initialize video pipeline.
        
        Args:
            model: PyTorch model for inference
            input_size: Input image size (H, W)
            batch_size: Batch size for inference
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.model.eval()
        self.input_size = input_size
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        
        if self.use_cuda:
            self.model = self.model.cuda()
        
        # Frame buffer for batching
        self.frame_buffer = deque(maxlen=batch_size)
        self.result_buffer = deque()
        
        # Statistics
        self.frame_count = 0
        self.total_inference_time = 0.0
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess video frame.
        
        Args:
            frame: Input frame (H, W, C) in BGR format
            
        Returns:
            Preprocessed tensor
        """
        # Resize
        frame = cv2.resize(frame, self.input_size)
        
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # HWC to CHW
        frame = np.transpose(frame, (2, 0, 1))
        
        return torch.from_numpy(frame)
    
    def process_batch(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Process a batch of frames.
        
        Args:
            frames: List of preprocessed frames
            
        Returns:
            List of predictions
        """
        # Stack frames into batch
        batch = torch.stack(frames)
        
        if self.use_cuda:
            batch = batch.cuda()
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(batch)
        
        if self.use_cuda:
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        self.total_inference_time += inference_time
        
        # Split batch into individual results
        results = [outputs[i] for i in range(len(frames))]
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Process a single frame with dynamic batching.
        
        Args:
            frame: Input video frame
            
        Returns:
            Prediction if batch is complete, None otherwise
        """
        # Preprocess
        preprocessed = self.preprocess_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(preprocessed)
        self.frame_count += 1
        
        # Process when buffer is full
        if len(self.frame_buffer) == self.batch_size:
            results = self.process_batch(list(self.frame_buffer))
            self.result_buffer.extend(results)
            self.frame_buffer.clear()
        
        # Return result if available
        if self.result_buffer:
            return self.result_buffer.popleft()
        
        return None
    
    def flush(self) -> List[torch.Tensor]:
        """
        Process remaining frames in buffer.
        
        Returns:
            List of remaining predictions
        """
        if len(self.frame_buffer) > 0:
            results = self.process_batch(list(self.frame_buffer))
            self.frame_buffer.clear()
            return results
        return []
    
    def get_stats(self) -> dict:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary of statistics
        """
        avg_inference_time = (
            self.total_inference_time / (self.frame_count / self.batch_size)
            if self.frame_count > 0 else 0
        )
        fps = self.batch_size / avg_inference_time if avg_inference_time > 0 else 0
        
        return {
            'frames_processed': self.frame_count,
            'avg_batch_time_ms': avg_inference_time * 1000,
            'fps': fps,
            'batch_size': self.batch_size
        }


class SimpleClassifier(nn.Module):
    """Simple classifier for demonstration."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def demonstrate_video_pipeline():
    """Demonstrate video pipeline optimization."""
    print("Video Pipeline Optimization")
    print("=" * 70)
    
    # Create model
    model = SimpleClassifier(num_classes=10)
    
    print("\n1. Pipeline Configuration")
    print(f"   Model: Simple CNN Classifier")
    print(f"   Input size: 224x224")
    print(f"   Batch sizes to test: [1, 2, 4, 8]")
    
    # Simulate video frames
    print("\n2. Generating Synthetic Video Frames")
    num_frames = 100
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(num_frames)]
    print(f"   Generated {num_frames} frames at 640x480")
    
    # Benchmark different batch sizes
    print("\n3. Batch Size Impact on Performance")
    print(f"{'Batch Size':<15} {'Total Time (s)':<20} {'FPS':<15} {'Latency (ms)':<20}")
    print("-" * 70)
    
    for batch_size in [1, 2, 4, 8]:
        pipeline = VideoPipeline(
            model,
            input_size=(224, 224),
            batch_size=batch_size,
            use_cuda=torch.cuda.is_available()
        )
        
        start_time = time.time()
        
        # Process frames
        for frame in frames:
            result = pipeline.process_frame(frame)
        
        # Flush remaining frames
        pipeline.flush()
        
        total_time = time.time() - start_time
        fps = num_frames / total_time
        latency = (total_time / num_frames) * 1000
        
        print(f"{batch_size:<15} {total_time:<20.3f} {fps:<15.2f} {latency:<20.2f}")
    
    # Best practices
    print("\n4. Optimization Techniques Applied")
    print("   ✓ Dynamic batching for throughput")
    print("   ✓ Efficient preprocessing pipeline")
    print("   ✓ GPU acceleration (if available)")
    print("   ✓ Frame buffer management")
    
    print("\n" + "=" * 70)
    print("Video Pipeline Best Practices:")
    print("1. Use dynamic batching to increase throughput")
    print("2. Optimize preprocessing (resize, normalize) on CPU")
    print("3. Use async data loading for zero-copy transfers")
    print("4. Consider model quantization for edge devices")
    print("5. Profile end-to-end latency vs throughput trade-offs")
    
    print("\nProduction Deployment:")
    print("- Use TensorRT for maximum GPU performance")
    print("- Implement multi-stream processing for concurrent videos")
    print("- Add frame skipping for real-time constraints")
    print("- Monitor GPU utilization and memory usage")


if __name__ == "__main__":
    demonstrate_video_pipeline()
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("1. Implement multi-threaded frame capture")
    print("2. Add GPU stream processing for parallelism")
    print("3. Integrate with video codecs (H.264, H.265)")
    print("4. Deploy with Triton Inference Server")
