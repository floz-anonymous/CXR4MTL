import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import random
from PIL import Image
import os
import cv2
import re
import math
from typing import Optional, Tuple
from collections import Counter
from transformers import BertTokenizer, BertModel 
from torchvision import transforms
from nltk.translate.bleu_score import corpus_bleu


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels: mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY, diffX = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, n_channels=3, bilinear=False):
        super(UNetEncoder, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1, self.down2, self.down3 = Down(64, 128), Down(128, 256), Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return [x1, x2, x3, x4, x5]

class UNetDecoder(nn.Module):
    def __init__(self, n_classes=1, bilinear=False):
        super(UNetDecoder, self).__init__()
        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2, self.up3, self.up4 = Up(512, 256 // factor, bilinear), Up(256, 128 // factor, bilinear), Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    def forward(self, encoder_features, mixed_features):
        x1, x2, x3, x4, _ = encoder_features
        x = self.up1(mixed_features, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class UNetPlusPlusDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=1, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        
        ch_0, ch_1, ch_2, ch_3, ch_4 = in_channels_list

        self.conv0_0 = DoubleConv(ch_0, 32, 32)
        self.conv1_0 = DoubleConv(ch_1, 64, 64)
        self.conv2_0 = DoubleConv(ch_2, 128, 128)
        self.conv3_0 = DoubleConv(ch_3, 256, 256)
        self.conv4_0 = DoubleConv(ch_4, 512, 512) 

        self.up1_0 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.up2_0 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.up3_0 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up4_0 = nn.ConvTranspose2d(512, 256, 2, 2) 

        self.up2_1 = nn.ConvTranspose2d(64, 32, 2, 2) 
        
        self.up3_1 = nn.ConvTranspose2d(128, 64, 2, 2) 
        
        self.up4_1 = nn.ConvTranspose2d(256, 128, 2, 2)
        
        self.up3_2 = nn.ConvTranspose2d(64, 32, 2, 2) 
        
        self.up4_2 = nn.ConvTranspose2d(128, 64, 2, 2)
        
        self.up4_3 = nn.ConvTranspose2d(64, 32, 2, 2) 

        self.conv0_1 = DoubleConv(32 + 32, 32, 32)
        self.conv1_1 = DoubleConv(64 + 64, 64, 64)
        self.conv2_1 = DoubleConv(128 + 128, 128, 128)
        self.conv3_1 = DoubleConv(256 + 256, 256, 256)

        self.conv0_2 = DoubleConv(32 + 32 + 32, 32, 32)
        self.conv1_2 = DoubleConv(64 + 64 + 64, 64, 64)
        self.conv2_2 = DoubleConv(128 + 128 + 128, 128, 128)

        self.conv0_3 = DoubleConv(32 + 32 + 32 + 32, 32, 32)
        self.conv1_3 = DoubleConv(64 + 64 + 64 + 64, 64, 64)

        self.conv0_4 = DoubleConv(32 + 32 + 32 + 32 + 32, 32, 32)
        
        self.final1 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final2 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final3 = nn.Conv2d(32, out_channels, kernel_size=1)
        self.final4 = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, encoder_features, mixed_features):
        x0_0, x1_0, x2_0, x3_0, _ = encoder_features 
        x4_0 = mixed_features 

        x0_0 = self.conv0_0(x0_0)
        x1_0 = self.conv1_0(x1_0)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up1_0(x1_0)], 1))
        
        x2_0 = self.conv2_0(x2_0)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up2_0(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up2_1(x1_1)], 1))

        x3_0 = self.conv3_0(x3_0)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up3_0(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up3_1(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up3_2(x1_2)], 1))

        x4_0 = self.conv4_0(mixed_features)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up4_0(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up4_1(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up4_2(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up4_3(x1_3)], 1))

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final4(x0_4)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x): return self.proj(x).flatten(2).transpose(1, 2)

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        x = (self.dropout(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SAMEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768, depth=8, num_heads=12, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim)) # +1 for CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        for block in self.blocks: x = block(x)
        return self.norm(x)

class EnhancedSummaryMixing(nn.Module):
    def __init__(self, local_dim=512, global_dim=768, output_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.local_proj = nn.Sequential(nn.Conv2d(local_dim, output_dim, kernel_size=1), nn.BatchNorm2d(output_dim), nn.ReLU())
        self.global_proj = nn.Sequential(nn.Linear(global_dim, output_dim), nn.LayerNorm(output_dim), nn.ReLU())
        self.fusion_conv = nn.Sequential(nn.Conv2d(output_dim * 2, output_dim, kernel_size=3, padding=1), nn.BatchNorm2d(output_dim), nn.ReLU())
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, local_features, global_features):
        B, C, H, W = local_features.shape
        local_proj = self.local_proj(local_features)
        global_proj = self.global_proj(global_features[:, 0, :])
        global_expanded = global_proj.unsqueeze(-1).unsqueeze(-1).expand(B, self.output_dim, H, W)
        combined = torch.cat([local_proj, global_expanded], dim=1)
        fused = self.fusion_conv(combined)
        return self.fusion_weight * local_proj + (1 - self.fusion_weight) * fused

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            
        self.attn = AttentionGate(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)
        
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x2_attn = self.attn(g=x1, x=x2)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
                        
        x = torch.cat([x2_attn, x1], dim=1)
        return self.conv(x)


class AttentionUNetDecoder(nn.Module):
    def __init__(self, n_classes=1, bilinear=False):
        super(AttentionUNetDecoder, self).__init__()
        factor = 2 if bilinear else 1
        
        self.up1 = AttentionUp(1024, 512 // factor, bilinear)
        self.up2 = AttentionUp(512, 256 // factor, bilinear)
        self.up3 = AttentionUp(256, 128 // factor, bilinear)
        self.up4 = AttentionUp(128, 64, bilinear)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, encoder_features, mixed_features):
        x1, x2, x3, x4, _ = encoder_features
        
        x = self.up1(mixed_features, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        return self.outc(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        channel_weights = self.channel_attention(x)
        x = x * channel_weights 
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights 
        return x

class AttentionClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.3):
        super(AttentionClassifierHead, self).__init__()
        
        self.attention = CBAMBlock(in_planes=input_dim)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.attention(x)
        
        x = self.global_pool(x)
        x = self.flatten(x)
        
        logits = self.classifier(x)
        
        return logits

class ClassifierHead(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        
        self.feature_attention = nn.MultiheadAttention(input_dim, num_heads=8, dropout=dropout, batch_first=True)
        self.feature_norm = nn.LayerNorm(input_dim)
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.residual_proj = nn.Linear(input_dim, input_dim * 2)
        
        self.pathway1 = nn.Sequential(
            nn.Linear(input_dim * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.pathway2 = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x_expanded = x.unsqueeze(1)  
        attended_x, _ = self.feature_attention(x_expanded, x_expanded, x_expanded)
        attended_x = self.feature_norm(attended_x.squeeze(1))
        
        enhanced_features = self.feature_extractor(attended_x)
        residual = self.residual_proj(x)
        enhanced_features = enhanced_features + residual
        
        path1_out = self.pathway1(enhanced_features)
        path2_out = self.pathway2(enhanced_features)
        
        fused = torch.cat([path1_out, path2_out], dim=1)
        logits = self.fusion(fused)
        
        return logits

class RRGDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout, max_len, input_feature_dim=1024, num_classes=10):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.image_proj = nn.Linear(input_feature_dim + embed_dim, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, image_features, target_reports, class_labels=None):
        batch_size, seq_len = target_reports.shape
        device = target_reports.device

        if class_labels is not None:
            label_embeds = self.label_embedding(class_labels)
            combined_features = torch.cat([image_features, label_embeds], dim=1)
        else:
            dummy_label_embeds = torch.zeros(batch_size, self.label_embedding.embedding_dim).to(device)
            combined_features = torch.cat([image_features, dummy_label_embeds], dim=1)

        image_memory = self.image_proj(combined_features).unsqueeze(1).expand(-1, seq_len, -1)
        embedded_reports = self.embedding(target_reports)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_embedding = self.position_embedding(positions)
        decoder_input = self.dropout(embedded_reports + pos_embedding)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        output = self.transformer_decoder(tgt=decoder_input, memory=image_memory, tgt_mask=tgt_mask)
        return self.fc_out(output)
    
    def generate_report(self, image_features, class_labels, tokenizer, max_length=100, temperature=1.0):
        device = image_features.device
        batch_size = image_features.size(0)
        
        generated = torch.full((batch_size, 1), tokenizer.cls_token_id, device=device, dtype=torch.long)
        
        if class_labels is not None:
            label_embeds = self.label_embedding(class_labels)
            combined_features = torch.cat([image_features, label_embeds], dim=1)
        else:
            dummy_label_embeds = torch.zeros(batch_size, self.label_embedding.embedding_dim).to(device)
            combined_features = torch.cat([image_features, dummy_label_embeds], dim=1)
        
        for step in range(max_length - 1):
            current_len = generated.size(1)
            
            image_memory = self.image_proj(combined_features).unsqueeze(1).expand(-1, current_len, -1)
            
            embedded = self.embedding(generated)
            positions = torch.arange(current_len, device=device).unsqueeze(0).expand(batch_size, -1)
            pos_embedding = self.position_embedding(positions)
            decoder_input = self.dropout(embedded + pos_embedding)
            
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len).to(device)
            
            output = self.transformer_decoder(tgt=decoder_input, memory=image_memory, tgt_mask=tgt_mask)
            logits = self.fc_out(output[:, -1, :]) 
            
            logits = logits / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if torch.all(next_token.squeeze() == tokenizer.sep_token_id):
                break
                
        return generated

class DualEncoderUNet(nn.Module):
    def __init__(self, input_channels, num_classes, img_size, patch_size, sam_embed_dim, sam_depth, sam_num_heads, bilinear, dropout):
        super().__init__()
        self.img_size = img_size
        self.unet_encoder = UNetEncoder(n_channels=input_channels, bilinear=bilinear)
        self.sam_encoder = SAMEncoder(img_size, patch_size, input_channels, sam_embed_dim, sam_depth, sam_num_heads, dropout=dropout)
        bottleneck_dim = 1024 if not bilinear else 512
        self.summary_mixing = EnhancedSummaryMixing(bottleneck_dim, sam_embed_dim, bottleneck_dim)
        # self.unet_decoder = UNetDecoder(n_classes=num_classes, bilinear=bilinear)
        # self.unet_decoder = AttentionUNetDecoder(n_classes=num_classes, bilinear=bilinear)
        self.unet_decoder = UNetPlusPlusDecoder(in_channels_list=[64, 128, 256, 512, 1024], out_channels=num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if x.shape[-2:] != (self.img_size, self.img_size):
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        unet_features = self.unet_encoder(x)
        sam_features = self.sam_encoder(x)
        mixed_features = self.summary_mixing(unet_features[-1], sam_features)
        seg_output = self.unet_decoder(unet_features, mixed_features)
        return seg_output, mixed_features, sam_features[:, 0, :] 


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class EnhancedDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, self_attn_mask):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.dropout(x) + residual

        residual = x
        x = self.cross_attn_layer_norm(x)
        x, _ = self.cross_attn(query=x, key=encoder_out, value=encoder_out)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.final_layer_norm(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x

class EnhancedRRGDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, dropout, max_len, 
                 input_feature_dim=1024, num_classes=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, dropout, max_len)
        
        self.feature_proj = nn.Sequential(
            nn.Linear(input_feature_dim + embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        self.layers = nn.ModuleList([
            EnhancedDecoderLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.embed_scale = math.sqrt(embed_dim)

    def _generate_causal_mask(self, sz, device):
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    def forward(self, image_features, target_reports, class_labels=None):
        batch_size, seq_len = target_reports.shape
        device = target_reports.device

        if class_labels is not None:
            label_embeds = self.label_embedding(class_labels)
            combined_features = torch.cat([image_features, label_embeds], dim=1)
        else:
            dummy_embeds = torch.zeros(batch_size, self.label_embedding.embedding_dim, device=device)
            combined_features = torch.cat([image_features, dummy_embeds], dim=1)
        
        encoder_out = self.feature_proj(combined_features).unsqueeze(1)

        token_embeds = self.token_embedding(target_reports) * self.embed_scale
        decoder_input = self.position_encoding(token_embeds)
        
        self_attn_mask = self._generate_causal_mask(seq_len, device)

        x = decoder_input
        for layer in self.layers:
            x = layer(x, encoder_out, self_attn_mask)
            
        return self.output_projection(x)

    def generate_report(self, image_features, class_labels, tokenizer, max_length=100, num_beams=4):
        self.eval() 
        device = image_features.device
        batch_size = image_features.size(0)

        if class_labels is not None:
            label_embeds = self.label_embedding(class_labels)
            combined_features = torch.cat([image_features, label_embeds], dim=1)
        else:
            dummy_embeds = torch.zeros(batch_size, self.label_embedding.embedding_dim, device=device)
            combined_features = torch.cat([image_features, dummy_embeds], dim=1)
        
        encoder_out = self.feature_proj(combined_features).unsqueeze(1)
        encoder_out = encoder_out.expand(-1, num_beams, -1).contiguous().view(batch_size * num_beams, 1, -1)

        input_ids = torch.full((batch_size * num_beams, 1), tokenizer.cls_token_id, device=device, dtype=torch.long)
        
        beam_scores = torch.zeros((batch_size, num_beams), device=device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)

        for _ in range(max_length - 1):
            token_embeds = self.token_embedding(input_ids) * self.embed_scale
            decoder_input = self.position_encoding(token_embeds)
            
            self_attn_mask = self._generate_causal_mask(input_ids.size(1), device)

            x = decoder_input
            for layer in self.layers:
                x = layer(x, encoder_out, self_attn_mask)
            
            logits = self.output_projection(x[:, -1, :])
            log_probs = F.log_softmax(logits, dim=-1)
            
            log_probs = log_probs + beam_scores.unsqueeze(1)
            log_probs = log_probs.view(batch_size, -1)
            
            beam_scores, beam_indices = torch.topk(log_probs, num_beams, dim=-1)

            beam_tokens = beam_indices % self.vocab_size
            beam_source_beam = (beam_indices // self.vocab_size).long()
            
            batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, num_beams)
            source_indices = (batch_idx * num_beams + beam_source_beam).view(-1)
            
            input_ids = torch.cat([input_ids[source_indices], beam_tokens.view(-1, 1)], dim=-1)
            beam_scores = beam_scores.view(-1)

            if torch.all(input_ids[:, -1] == tokenizer.sep_token_id):
                break
        
        best_beam_idx = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
        final_sequences = []
        for i in range(batch_size):
            best_seq_for_sample = input_ids[i * num_beams + best_beam_idx[i]]
            final_sequences.append(best_seq_for_sample)

        return torch.stack(final_sequences)


class UnifiedMultiTaskModel(nn.Module):
    def __init__(self, seg_n_classes, class_n_classes, rrg_vocab_size, vqa_vocab_size, args):
        super().__init__()
        self.encoder = DualEncoderUNet(3, seg_n_classes, args.img_size, 16, 768, 6, 12, False, args.dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # self.classifier_head = ClassifierHead(input_dim=1024, num_classes=class_n_classes, dropout=args.dropout)
        self.classifier_head = AttentionClassifierHead(input_dim=1024, num_classes=class_n_classes, dropout=args.dropout)

        # self.rrg_decoder = RRGDecoder(
        #     rrg_vocab_size, args.embed_dim, args.num_layers, 12, args.dropout, args.max_len, 1024, num_classes=class_n_classes
        # )
        self.rrg_decoder = EnhancedRRGDecoder(
            vocab_size=rrg_vocab_size, 
            embed_dim=args.embed_dim, 
            num_layers=args.num_layers, 
            num_heads=8,  # A common choice for decoder heads
            dropout=args.dropout, 
            max_len=args.max_len, 
            input_feature_dim=1024,
            num_classes=class_n_classes
        )

    def forward(self, x, mode, rrg_targets=None, class_labels=None, question_ids=None, question_masks=None, answer_ids=None):
        seg_output, features, sam_cls_feature = self.encoder(x)

        if mode == 'seg':
            return seg_output, features

        pooled_features = self.global_pool(features)
        flattened_features = self.flatten(pooled_features)

        if mode == 'class':
            # logits = self.classifier_head(flattened_features)
            logits = self.classifier_head(features) # for attention
            return F.softmax(logits, dim=1), logits

        elif mode == 'rrg':
            if rrg_targets is None:
                raise ValueError("rrg_targets must be provided for RRG mode during training.")
            output = self.rrg_decoder(flattened_features, target_reports=rrg_targets, class_labels=class_labels)
            return output

        else:
            raise ValueError(f"Unknown mode: {mode}")