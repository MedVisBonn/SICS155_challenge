"""
ResNet Backbone model for image classification with custom prediction OkO-head

Based on ResNet implementation of torchvision and OkO head implementation of
# TODO: add citation to OkO model!!
"""

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision.models as models


@dataclass
class BackBoneConfig:
    """Configuration class for ResNet backbone."""
    backbone_type: str = "resnet"
    variant: str = "50"  # 18, 34, 50, 101, 152
    num_classes_phase: int = 19
    num_classes_tools: Optional[int] = None
    pretrained: bool = True
    multi_task: bool = False
    oko_k: Optional[int] = None
    extended_dim: Optional[int] = None


class PhaseHead(nn.Module):
    """Phase classification head."""
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class ToolHead(nn.Module):
    """Tool classification head."""
    
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(in_features, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(x)


class OKOHead(nn.Module):
    """Odd-K-Odd (OKO) classification head."""
    
    def __init__(self, in_features: int, num_classes: int, k: int, backbone_type: str = "resnet"):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.k = k
        self.dropout = 0.5
        self.backbone_type = backbone_type
        self.register_parameter(
            "reweighting", nn.Parameter(torch.ones((self.k + 2) * self.in_features))
        )
        self._build_head()
        
    def _build_head(self) -> None:
        """Build query and key networks based on backbone type."""
        if self.backbone_type.startswith(("resnet", "vit", "endovit")):
            self.query = nn.Linear(self.in_features, self.num_classes, bias=True)
            self.key = nn.Linear(
                (self.k + 2) * self.in_features, self.num_classes, bias=True
            )
        elif self.backbone_type.startswith("efficientnet"):
            self.query = nn.Sequential(
                nn.Dropout(p=self.dropout, inplace=False),
                nn.Linear(self.in_features, self.num_classes, bias=True),
            )
            self.key = nn.Sequential(
                nn.Dropout(p=self.dropout, inplace=False),
                nn.Linear((self.k + 2) * self.in_features, self.num_classes, bias=True),
            )
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
    
    def forward(self, x: Tensor, train: bool = True) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for OKO head.
        
        Args:
            x: Input tensor of shape (B, C)
            train: Whether in training mode
            
        Returns:
            Tuple of (query output, key output)
        """
        if not train:
            return self.query(x), None
        
        # Set up batch indices for each part (p: positive, n: negative, etc.)
        batch_size = x.shape[0] // (self.k + 2)
        
        # Split the batch into positive, negative, anchor and neighbors
        batch_p = x[:batch_size]  # Positive
        batch_n = x[batch_size : 2 * batch_size]  # Negative
        
        # Rest are neighbors for the anchor
        features_neighbors = x[2 * batch_size :]
        
        # Group neighbors for each anchor
        # This creates a tensor of shape (batch_size, k * feature_dim)
        features_neighbors = features_neighbors.reshape(batch_size, self.k * self.in_features)
        
        # Prepare the key input by concatenating p, n, and neighbors
        key_input = torch.cat([batch_p, batch_n, features_neighbors], dim=1)
        
        # Get predictions
        out_p = self.query(batch_p)
        out_n = self.query(batch_n)
        out_key = self.key(key_input)
        
        return out_p, out_n
    

class FeatureTransform(nn.Module):
    """Feature transformation module to modify feature dimension."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.transform = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Flatten(),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features, out_features),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transform(x)


class ResNetBackbone(nn.Module):
    """ResNet backbone implementation.

    Attributes:
        SUPPORTED_VARIANTS: Mapping of variant names to model constructors
        variant: Selected ResNet variant (e.g., 18, 50, 152)
        backbone: Feature extraction backbone
        heads: Dict of classification heads
    """

    SUPPORTED_VARIANTS = {
        "18": models.resnet18,
        "34": models.resnet34,
        "50": models.resnet50,
        "101": models.resnet101,
        "152": models.resnet152,
    }

    def __init__(self, config: BackBoneConfig):
        """Initialize ResNet backbone.

        Args:
            config: Configuration object containing model parameters

        Raises:
            ValueError: If unsupported variant is specified
        """
        super().__init__()
        self.config = config
        
        if config.variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported ResNet variant: {config.variant}. "
                f"Supported variants are: {list(self.SUPPORTED_VARIANTS.keys())}"
            )
        
        # Initialize feature dimension
        if config.variant in ["18", "34"]:
            self.feature_dim = 512
        else:
            self.feature_dim = 2048
            
        # Initialize backbone
        self._init_backbone()
        
        # Initialize heads
        self.heads = self._init_heads()
        
    def _init_backbone(self) -> None:
        """Initialize ResNet backbone"""
        weights = "DEFAULT" if self.config.pretrained else None
        model = self.SUPPORTED_VARIANTS[self.config.variant](weights=weights)

        # Set the classification layer to Identity
        model.fc = nn.Identity()
        
        self.backbone = model

    def _init_heads(self) -> nn.ModuleDict:
        """Initialize classification heads."""
        heads = nn.ModuleDict()

        # Feature transformation if needed
        if self.config.extended_dim is not None:
            # Add additional head to extend the feature vector resolution/dimension
            heads["feature"] = FeatureTransform(
                in_features=self.feature_dim,
                out_features=self.config.extended_dim,
            )
            feat_dim = self.config.extended_dim
        else:
            feat_dim = self.feature_dim

        # Phase classification head (OKO or standard)
        if self.config.oko_k is not None:
            oko_head = OKOHead(
                in_features=feat_dim,
                num_classes=self.config.num_classes_phase,
                k=self.config.oko_k,
                backbone_type=self.config.backbone_type,
            )
            heads["phase"] = oko_head
        else:
            phase_head = PhaseHead(feat_dim, self.config.num_classes_phase)
            heads["phase"] = phase_head

        # Tool classification head (if multi-task)
        if self.config.multi_task and self.config.num_classes_tools:
            tool_head = ToolHead(feat_dim, self.config.num_classes_tools)
            heads["tool"] = tool_head

        return heads

    def forward(
        self, x: Tensor, train: bool = True
    ) -> Union[
        Tuple[Tensor, Tensor],
        Tuple[Tensor, Tuple[Tensor, Tensor]],
        Tuple[Tensor, Tensor, Tensor],
    ]:
        """Forward pass through the backbone and classification heads.

        Args:
            x: Input tensor of shape (B, C, H, W)
            train: Whether in training mode

        Returns:
            Tuple containing:
                - Feature tensor (B, feature_dim)
                - Phase predictions or OKO outputs
                - Tool predictions (if multi_task=True)
        """
        features = self.backbone(x)
        outputs = [features]

        if "feature" in self.heads:
            features = self.heads["feature"](features)
            outputs = [features]

        if self.config.oko_k is not None:
            # Handle OKO training mode differently
            phase_out = self.heads["phase"](features, train=train)
        else:
            phase_out = self.heads["phase"](features)
        outputs.append(phase_out)

        if "tool" in self.heads:
            tool_out = self.heads["tool"](features)
            outputs.append(tool_out)

        return tuple(outputs)


class ResNetModel:
    """Wrapper class for ResNet model to be used in challenge submissions."""
    
    def __init__(self, config: Optional[BackBoneConfig] = None, **kwargs):
        """Initialize ResNet model.
        
        Args:
            config: BackBoneConfig object
            **kwargs: Alternative configuration as keyword arguments
        """
        # Create config if not provided
        if config is None:
            config = BackBoneConfig(
                backbone_type=kwargs.get("backbone_type", "resnet"),
                variant=kwargs.get("variant", "50"),
                num_classes_phase=kwargs.get("num_classes_phase", 19),
                num_classes_tools=kwargs.get("num_classes_tools"),
                pretrained=kwargs.get("pretrained", True),
                multi_task=kwargs.get("multi_task", False),
                oko_k=kwargs.get("oko_k"),
                extended_dim=kwargs.get("extended_dim"),
            )
        
        self.model = ResNetBackbone(config)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for inference.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            Class predictions
        """
        with torch.no_grad():
            outputs = self.model(x, train=False)
            
            # Handle different output formats
            if isinstance(outputs[1], tuple):  # OKO output
                return outputs[1][0]  # Return only the primary output
            else:
                return outputs[1]  # Regular output
    
    def extract_features(self, x: Tensor) -> Tensor:
        """Extract features from the backbone without classification.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor (batch_size, feature_dim)
        """
        with torch.no_grad():
            # Ensure input is on the same device as the model
            device = next(self.model.backbone.parameters()).device
            x = x.to(device)
            
            # Only run the backbone part of the model
            features = self.model.backbone(x)
            
            # Apply feature transformation if available
            if "feature" in self.model.heads:
                features = self.model.heads["feature"](features)
                
            return features
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        """Load a model from a Lightning checkpoint."""
        # Create a new instance
        model = cls(**kwargs)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get state dict from Lightning checkpoint
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # Print key structure to help debugging
            print(f"Checkpoint keys: {list(state_dict.keys())[:5]} (showing first 5 keys)")
            
            # Create new state dict with appropriate prefixes removed
            new_state_dict = {}
            for k, v in state_dict.items():
                # Handle different Lightning module key patterns
                if k.startswith('backbone.'):
                    new_state_dict[k.replace('backbone.', '')] = v
                elif k.startswith('model.'):
                    new_state_dict[k.replace('model.', '')] = v
                else:
                    # Keep as is for other keys
                    new_state_dict[k] = v
            
            # Try loading the state dict
            try:
                model.model.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"First loading attempt failed: {e}")
                print("Trying alternative loading approach...")
                
                # Try direct loading
                try:
                    model.model.load_state_dict(state_dict)
                except Exception as e2:
                    print(f"Second loading attempt failed: {e2}")
                    # One more attempt with stripped model prefix
                    try:
                        stripped_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
                        model.model.load_state_dict(stripped_dict)
                    except Exception as e3:
                        print(f"All loading attempts failed, last error: {e3}")
                        raise
        else:
            # Direct checkpoint (not from Lightning)
            model.model.load_state_dict(checkpoint)
        
        # Set to evaluation mode
        model.model.eval()
        
        return model


def create_resnet_model(variant="50", num_classes=19, oko_k=None, checkpoint_path=None):
    """Create a ResNet model for challenge submission.
    
    Args:
        variant: ResNet variant (18, 34, 50, 101, 152)
        num_classes: Number of output classes
        oko_k: OKO parameter k (None for standard model)
        checkpoint_path: Path to trained model checkpoint or standalone state dict
    
    Returns:
        ResNet model ready for inference
    """
    config = BackBoneConfig(
        backbone_type="resnet",
        variant=variant,
        num_classes_phase=num_classes,
        pretrained=True if not checkpoint_path else False,
        oko_k=oko_k
    )
    
    # First create the model
    model = ResNetModel(config=config)
    
    # Then load weights if provided
    if checkpoint_path:
        try:
            # Try loading as a standalone state dict first
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if it's a state dict directly or has a state_dict key
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    # It's a Lightning checkpoint
                    model = ResNetModel.load_from_checkpoint(
                        checkpoint_path,
                        config=config
                    )
                else:
                    # It's a standalone state dict
                    model.model.load_state_dict(state_dict)
                    model.model.eval()
            else:
                # It's something else, try ResNetModel.load_from_checkpoint
                model = ResNetModel.load_from_checkpoint(
                    checkpoint_path,
                    config=config
                )
                
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Using default pretrained weights instead.")
    
    return model
