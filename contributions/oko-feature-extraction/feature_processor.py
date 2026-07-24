import numpy as np
import torch
from tqdm import tqdm
import argparse
import os

# from backbone_model import BackBoneModel
from temporal_model import create_asformer_model
from backbone_model import create_resnet_model
from extract_features import extract_and_preprocess_video


class Processor:
    def __init__(self):
        # Check if GPU is available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize backbone model for feature extraction
        print("Initializing backbone model...")
        self.backbone = create_resnet_model(
            variant="50",
            num_classes=19,
            oko_k=1,
            # checkpoint_path="/app/ingested_program/standalone_backbone_weights.pth",
            checkpoint_path="standalone_backbone_weights.pth",
        )
        self.backbone.model.eval()  # Set to evaluation mode
        self.backbone.model.to(self.device)  # Move model to GPU
        self.name = "OkO-ASFormer"

    def fit(self):
        """Train model here -> Not Needed"""
        pass

    def process_features(self, video_paths, output_dir):
        """Predict phase labels from features"""
        predictions = {}
        
        # Don't use the provided I3D features
        # for key, feature in features.items():
            # predictions[key] = self.classifier.predict(feature)
        
        for key, path in video_paths.items():
            print(f"## Processing video: {key} ##")
            
            # Extract and preprocess frames in one step
            import time
            start_time = time.time()
            
            # This will return a proper torch.Tensor ready for the model
            # Only use GPU if device is CUDA
            use_gpu_for_extraction = str(self.device).startswith('cuda')
            video_frames = extract_and_preprocess_video(
                path,
                fps=15,
                use_gpu=False,
                target_size=(224, 224),
                batch=True
            )

            # TODO: Don't move at once! move inside the batches
            # video_frames = video_frames.to(self.device)
            
            end_time = time.time()
            print(f"-- Video loading and preprocessing time: {end_time - start_time:.2f} seconds")
            # print(f"Frames tensor shape: {video_frames.shape}")

            start_time = time.time()
            # Extract backbone features in batches and collect them
            batch_size = 32  # Process frames in batches to avoid OOM
            backbone_features_list = []
            
            with torch.no_grad():
                total_batches = (len(video_frames) + batch_size - 1) // batch_size
                for i in tqdm(range(0, len(video_frames), batch_size), 
                              total=total_batches, 
                              desc="Extracting features", 
                              unit="batch"):
                    batch = video_frames[i:i + batch_size]
                    # Move batch to GPU
                    batch = batch.to(self.device)
                    # Extract features from the backbone (before classification head)
                    features = self.backbone.extract_features(batch)
                    backbone_features_list.append(features)
                    
                    # Explicitly clear memory
                    del features
                    del batch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Combine all batched features
            backbone_features = torch.cat(backbone_features_list, dim=0)
            # Ensure features are on GPU (they should already be, but just to be safe)
            backbone_features = backbone_features.detach().cpu().numpy()
            np.save(os.path.join(output_dir, f"{key}_oko.npy"), backbone_features)

            end_time = time.time()
            print(f"-- Backbone feature extraction time: {end_time - start_time:.2f} seconds")

           
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from videos")
    parser.add_argument("--video_dir", type=str, help="Directory containing videos", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory to save extracted features", required=True)
    args = parser.parse_args()

    videos = os.listdir(args.video_dir)
    processor = Processor()
    video_paths = {vid: os.path.join(args.video_dir, vid) for vid in videos}
    os.makedirs(args.output_dir, exist_ok=True)
    features = processor.extract_features(video_paths, args.output_dir)