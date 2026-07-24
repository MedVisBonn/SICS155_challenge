"""
ASFormer: Action Segmentation with Transformer model.

Based on: https://github.com/ChinaYi/ASFormer
"""

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


@dataclass
class ASFormerConfig:
    """Configuration class for ASFormer model."""
    num_decoders: int
    num_layers: int
    r1: int
    r2: int
    num_f_maps: int
    input_dim: int
    num_classes: int
    channel_masking_rate: float


def exponential_descrease(idx_decoder, p=3):
    """Calculate exponential decrease factor for decoder layers."""
    return math.exp(-p * idx_decoder)


class AttentionHelper(nn.Module):
    """Helper class for calculating attention."""
    
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        """Calculate scalar dot attention."""
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape

        assert c1 == c2

        energy = torch.bmm(
            proj_query.permute(0, 2, 1), proj_key
        )  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(
            padding_mask + 1e-6
        )  # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention)
        attention = attention * padding_mask
        attention = attention.permute(0, 2, 1)
        out = torch.bmm(proj_val, attention)
        return out, attention


class AttLayer(nn.Module):
    """Attention layer for the ASFormer model."""
    
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type):
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)
        
        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ["normal_att", "block_att", "sliding_att"]
        assert self.stage in ["encoder", "decoder"]
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
    
    def construct_window_mask(self):
        """Construct window mask for sliding window attention."""
        window_mask = torch.zeros((1, self.bl, self.bl + 2 * (self.bl // 2)))
        for i in range(self.bl):
            window_mask[:, :, i : i + self.bl] = 1
        return window_mask
    
    def forward(self, x1, x2, mask):
        """Forward pass through attention layer."""
        # x1 from the encoder, x2 from the decoder
        query = self.query_conv(x1)
        key = self.key_conv(x1)
        
        if self.stage == "decoder":
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
        
        if self.att_type == "normal_att":
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == "block_att":
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == "sliding_att":
            return self._sliding_window_self_att(query, key, value, mask)
    
    def _normal_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()

        device = q.device
        padding_mask = torch.ones((m_batchsize, 1, L), device=device) * mask[:, 0:1, :]

        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]
    
    def _block_wise_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, L = k.size()
        _, c3, L = v.size()
        
        device = q.device
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl), device=device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl), device=device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl), device=device)], dim=-1)
            nb += 1
        
        padding_mask = torch.cat([
            torch.ones((m_batchsize, 1, L), device=device) * mask[:, 0:1, :],
            torch.zeros((m_batchsize, 1, self.bl * nb - L), device=device)
        ], dim=-1)
        
        # Reshape for block attention
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, 1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]
    
    def _sliding_window_self_att(self, q, k, v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        device = q.device
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl), device=device, dtype=q.dtype)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl), device=device, dtype=k.dtype)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl), device=device, dtype=v.dtype)], dim=-1)
            nb += 1
        
        padding_mask = torch.cat([
            torch.ones((m_batchsize, 1, L), device=device, dtype=mask.dtype) * mask[:, 0:1, :],
            torch.zeros((m_batchsize, 1, self.bl * nb - L), device=device, dtype=mask.dtype)
        ], dim=-1)
        
        # sliding window approach for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([
            torch.zeros(m_batchsize, c2, self.bl // 2, device=device, dtype=k.dtype),
            k,
            torch.zeros(m_batchsize, c2, self.bl // 2, device=device, dtype=k.dtype)
        ], dim=-1)
        v = torch.cat([
            torch.zeros(m_batchsize, c3, self.bl // 2, device=device, dtype=v.dtype),
            v,
            torch.zeros(m_batchsize, c3, self.bl // 2, device=device, dtype=v.dtype)
        ], dim=-1)
        padding_mask = torch.cat([
            torch.zeros(m_batchsize, 1, self.bl // 2, device=device, dtype=mask.dtype),
            padding_mask,
            torch.zeros(m_batchsize, 1, self.bl // 2, device=device, dtype=mask.dtype)
        ], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)], dim=0)
        v = torch.cat([v[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2] for i in range(nb)], dim=0)
        padding_mask = torch.cat(
                    [
                        padding_mask[:, :, i * self.bl : (i + 1) * self.bl + (self.bl // 2) * 2]
                        for i in range(nb)
                    ],
                    dim=0,
                )
        
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        final_mask = self.window_mask.to(device=device, dtype=mask.dtype).repeat(m_batchsize * nb, 1, 1) * padding_mask
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    """Multi-head attention layer."""
    
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList([
            copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type))
            for i in range(num_head)
        ])
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out


class ConvFeedForward(nn.Module):
    """Convolutional feed forward layer."""
    
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    """Fully connected feed forward layer."""
    
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
    
    def forward(self, x):
        return self.layer(x)


class AttModule(nn.Module):
    """Attention module combining attention and feed-forward layers."""
    
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(
            in_channels,
            in_channels,
            out_channels,
            r1, r1, r2,
            dilation,
            att_type=att_type,
            stage=stage
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
    
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0, 2, 1)  # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
    
    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]


class Encoder(nn.Module):
    """Encoder module for ASFormer."""
    
    def __init__(
        self,
        num_layers,
        r1,
        r2,
        num_f_maps,
        input_dim,
        num_classes,
        channel_masking_rate,
        att_type,
        alpha,
    ):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)  # fc layer
        self.layers = nn.ModuleList([
            AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "encoder", alpha)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
    
    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)
        
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]
        
        return out, feature


class Decoder(nn.Module):
    """Decoder module for ASFormer."""
    
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList([
            AttModule(2**i, num_f_maps, num_f_maps, r1, r2, att_type, "decoder", alpha)
            for i in range(num_layers)
        ])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
    
    def forward(self, x, fencoder, mask):
        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]
        
        return out, feature


class ASFormer(nn.Module):
    """Action Segmentation with Transformer model."""
    
    def __init__(self, config: ASFormerConfig):
        super(ASFormer, self).__init__()
        self.config = config
        self.encoder = Encoder(
            self.config.num_layers,
            self.config.r1,
            self.config.r2,
            self.config.num_f_maps,
            self.config.input_dim,
            self.config.num_classes,
            self.config.channel_masking_rate,
            att_type="sliding_att",
            alpha=1,
        )
        self.decoders = nn.ModuleList([
            copy.deepcopy(Decoder(
                self.config.num_layers,
                self.config.r1,
                self.config.r2,
                self.config.num_f_maps,
                self.config.num_classes,
                self.config.num_classes,
                att_type="sliding_att",
                alpha=exponential_descrease(s),
            ))
            for s in range(self.config.num_decoders)
        ])
    
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], 
                                  feature * mask[:, 0:1, :], 
                                  mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        
        return feature, outputs


class TemporalModel(nn.Module):
    """Wrapper class for ASFormer model to be used in challenge submissions."""
    
    def __init__(self, config: Optional[ASFormerConfig] = None, **kwargs):
        """Initialize a temporal model.
        
        Args:
            config: ASFormerConfig object
            **kwargs: Alternative configuration as keyword arguments
        """
        super(TemporalModel, self).__init__()
        
        # Create config if not provided
        if config is None:
            config = ASFormerConfig(
                num_decoders=kwargs.get("num_decoders", 3),
                num_layers=kwargs.get("num_layers", 10),
                r1=kwargs.get("r1", 2),
                r2=kwargs.get("r2", 2),
                num_f_maps=kwargs.get("num_f_maps", 64),
                input_dim=kwargs.get("input_dim", 2048),
                num_classes=kwargs.get("num_classes", 19),
                channel_masking_rate=kwargs.get("channel_masking_rate", 0.3),
            )
        
        self.model = ASFormer(config)
    
    def forward(self, x):
        """Forward pass for inference.
        
        Args:
            x: Input features tensor of shape (batch_size, channels, sequence_length)
        
        Returns:
            Predicted class probabilities of shape (batch_size, num_classes, sequence_length)
        """
        # Ensure x is on the correct device
        device = next(self.model.parameters()).device
        x = x.to(device)
        
        # Create a mask (all ones) on the same device
        mask = torch.ones(x.size(0), 1, x.size(2), device=device)
        
        # Get predictions (feature, outputs)
        _, outputs = self.model(x, mask)
        
        # Use the final decoder output
        logits = outputs[-1]
        
        return logits
    
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


def create_asformer_model(input_dim=2048, num_classes=19, checkpoint_path=None):
    """Create an ASFormer model for challenge submission.
    
    Args:
        input_dim: Dimension of input features
        num_classes: Number of output classes
        checkpoint_path: Path to trained model checkpoint
    
    Returns:
        ASFormer model ready for inference
    """
    config = ASFormerConfig(
        num_decoders=3,
        num_layers=10,
        r1=2,
        r2=2,
        num_f_maps=64,
        input_dim=input_dim,
        num_classes=num_classes,
        channel_masking_rate=0.0,  # Set to 0 for inference
    )
    
    if checkpoint_path:
        model = TemporalModel.load_from_checkpoint(
            checkpoint_path, 
            config=config
        )
    else:
        model = TemporalModel(config=config)
    
    return model
