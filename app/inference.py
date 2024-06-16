# inference.py

import argparse
import os
import sys
from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '/app'))

import torch
from tqdm import tqdm
from hallo.animate.face_animate import FaceAnimatePipeline
from config import load_config
from audio_processing import AudioProcessing, process_audio_emb
from image_processing import ImageProcessing
from models import build_modules
from hallo.utils.util import tensor_to_video


def inference_process(args):
    logger.info("Loading configuration from {}", args.config)
    config = load_config(args.config, args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32  # Default weight data type

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif config.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    elif config.weight_dtype == "fp32":
        weight_dtype = torch.float32

    logger.info("Initializing image processing components")
    image_processing = ImageProcessing(config)
    source_image_pixels, source_image_face_region, source_image_face_emb, \
    source_image_full_mask, source_image_face_mask, source_image_lip_mask = image_processing.preprocess(
        config.source_image, config.save_path)

    logger.info("Initializing audio processing components")
    audio_processing = AudioProcessing(config)
    audio_emb = audio_processing.preprocess(config.driving_audio, config.save_path)

    logger.info("Building model modules")
    net, vae, val_noise_scheduler = build_modules(config, device, weight_dtype)

    pipeline = FaceAnimatePipeline(
        vae=vae, reference_unet=net.reference_unet, denoising_unet=net.denoising_unet,
        face_locator=net.face_locator, scheduler=val_noise_scheduler, image_proj=net.imageproj,
    )
    pipeline.to(device=device, dtype=weight_dtype)

    audio_emb = process_audio_emb(audio_emb)
    source_image_pixels = source_image_pixels.unsqueeze(0)
    source_image_face_region = source_image_face_region.unsqueeze(0)
    source_image_face_emb = source_image_face_emb.reshape(1, -1)
    source_image_face_emb = torch.tensor(source_image_face_emb)

    logger.info("Preparing image masks for animation")
    source_image_full_mask, source_image_face_mask, source_image_lip_mask = [
        [mask.repeat(config.data.n_sample_frames, 1) for mask in mask_group]
        for mask_group in [source_image_full_mask, source_image_face_mask, source_image_lip_mask]
    ]

    times = audio_emb.shape[0] // config.data.n_sample_frames
    tensor_result = []
    generator = torch.manual_seed(42)

    for t in tqdm(range(times), desc="Processing"):
        if len(tensor_result) == 0:
            motion_zeros = source_image_pixels.repeat(config.data.n_motion_frames, 1, 1, 1)
            motion_zeros = motion_zeros.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_zeros], dim=0)
        else:
            motion_frames = tensor_result[-1][0]
            motion_frames = motion_frames.permute(1, 0, 2, 3)
            motion_frames = motion_frames[0-config.data.n_motion_frames:]
            motion_frames = motion_frames * 2.0 - 1.0
            motion_frames = motion_frames.to(dtype=source_image_pixels.dtype, device=source_image_pixels.device)
            pixel_values_ref_img = torch.cat([source_image_pixels, motion_frames], dim=0)

        pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
        audio_tensor = audio_emb[t * config.data.n_sample_frames: min((t + 1) * config.data.n_sample_frames, audio_emb.shape[0])]
        audio_tensor = audio_tensor.unsqueeze(0).to(device=net.audioproj.device, dtype=net.audioproj.dtype)
        audio_tensor = net.audioproj(audio_tensor)

        logger.info("Animating frame {}", t)
        pipeline_output = pipeline(
            ref_image=pixel_values_ref_img, audio_tensor=audio_tensor, face_emb=source_image_face_emb,
            face_mask=source_image_face_region, pixel_values_full_mask=source_image_full_mask,
            pixel_values_face_mask=source_image_face_mask, pixel_values_lip_mask=source_image_lip_mask,
            width=config.data.source_image.width, height=config.data.source_image.height,
            video_length=config.data.n_sample_frames, num_inference_steps=config.inference_steps,
            guidance_scale=config.cfg_scale, generator=generator,
            motion_scale=[config.pose_weight, config.face_weight, config.lip_weight],
        )

        tensor_result.append(pipeline_output.videos)

    tensor_result = torch.cat(tensor_result, dim=2)
    tensor_result = tensor_result.squeeze(0)
    logger.info("Generating final video")
    tensor_to_video(tensor_result, config.output, config.driving_audio)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default="configs/inference/default.yaml")
    parser.add_argument("--source_image", type=str, default="./HumanAssets/Data/man3.png")
    parser.add_argument("--driving_audio", type=str, default="./HumanAssets/Data/aud3.mp3")
    parser.add_argument("--output", type=str, default=".cache/output.mp4")
    parser.add_argument("--pose_weight", type=float, default=1.0)
    parser.add_argument("--face_weight", type=float, default=1.0)
    parser.add_argument("--lip_weight", type=float, default=1.0)
    parser.add_argument("--face_expand_ratio", type=float, default=1.2)
    parser.add_argument("--checkpoint", type=str, default=None)

    command_line_args = parser.parse_args()
    inference_process(command_line_args)

if __name__ == "__main__":
    main()
