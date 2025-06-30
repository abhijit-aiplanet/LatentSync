import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
import os

# A100 optimized configurations
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")  # Use 512x512 for A100
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

# Set A100 specific optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    use_deepcache,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Set the output path for the processed video
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")

    config = OmegaConf.load(CONFIG_PATH)

    config["run"].update(
        {
            "guidance_scale": guidance_scale,
            "inference_steps": inference_steps,
        }
    )

    # Parse the arguments
    args = create_args(video_path, audio_path, output_path, inference_steps, guidance_scale, seed, use_deepcache)

    try:
        # Clear GPU cache before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        result = main(
            config=config,
            args=args,
        )
        print("Processing completed successfully.")
        return output_path
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, 
    guidance_scale: float, seed: int, use_deepcache: bool
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    args_list = [
        "--inference_ckpt_path", CHECKPOINT_PATH.absolute().as_posix(),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--temp_dir", "temp",
    ]
    
    if use_deepcache:
        args_list.append("--enable_deepcache")

    return parser.parse_args(args_list)


def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"🚀 Running on: {gpu_name} ({gpu_memory:.1f}GB VRAM)"
    return "❌ No GPU detected"


# Create Gradio interface with A100 optimizations
with gr.Blocks(title="LatentSync A100 Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
    <h1 align="center">🚀 LatentSync A100 Optimized</h1>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <p><strong>{get_gpu_info()}</strong></p>
        <p>High-resolution lip-sync with 512×512 processing</p>
    </div>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/bytedance/LatentSync">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a> 
        <a href="https://arxiv.org/abs/2412.09262">
            <img src='https://img.shields.io/badge/arXiv-Paper-red'>
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📹 Input")
            video_input = gr.Video(label="Input Video", height=300)
            audio_input = gr.Audio(label="Input Audio", type="filepath")

            gr.Markdown("### ⚙️ Settings")
            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,  # Optimized for A100
                    step=0.1,
                    label="Guidance Scale",
                    info="Higher = better sync, but may cause artifacts"
                )
                inference_steps = gr.Slider(
                    minimum=15, 
                    maximum=50, 
                    value=25,  # Higher default for A100
                    step=1, 
                    label="Inference Steps",
                    info="More steps = better quality, slower processing"
                )

            with gr.Row():
                seed = gr.Number(
                    value=1247, 
                    label="Random Seed", 
                    precision=0,
                    info="Set to -1 for random"
                )
                use_deepcache = gr.Checkbox(
                    value=True,  # Enable by default for A100
                    label="Enable DeepCache",
                    info="Faster processing with minimal quality loss"
                )

            process_btn = gr.Button("🎬 Process Video", variant="primary", size="lg")
            
            gr.Markdown("### 💡 Tips for A100")
            gr.Markdown(
                """
                - **Guidance Scale 2.0-2.5**: Best balance of quality and stability
                - **25-35 Steps**: Optimal for high-quality results
                - **DeepCache**: Recommended for faster processing
                - **512×512 Resolution**: Maximum quality on A100
                """
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Output")
            video_output = gr.Video(label="Lip-synced Video", height=400)
            
            gr.Markdown("### 📁 Example Files")
            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
                label="Try these examples:"
            )

    # Processing function
    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
            use_deepcache,
        ],
        outputs=video_output,
        show_progress=True,
    )

if __name__ == "__main__":
    print("🚀 Starting LatentSync A100 Gradio Interface...")
    print(f"🎯 {get_gpu_info()}")
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=True,  # Create public URL
        inbrowser=False,  # Don't try to open browser in RunPod
        show_error=True,
        favicon_path=None
    ) 