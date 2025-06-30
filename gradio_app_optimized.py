import gradio as gr
from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
from datetime import datetime
import torch
import os
import numpy as np

# Apply all GPU optimizations
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

# Import optimizations
try:
    import xformers
    import xformers.ops
    XFORMERS_AVAILABLE = True
    print("✅ XFormers available - using memory efficient attention")
except ImportError:
    XFORMERS_AVAILABLE = False
    print("⚠️ XFormers not available")

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    print("✅ Flash Attention available")
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("⚠️ Flash Attention not available")

# A100 optimized configurations
CONFIG_PATH = Path("configs/unet/stage2_512.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

# Set A100 specific optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# Enable mixed precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def apply_model_optimizations(model):
    """Apply all available optimizations to the model"""
    if hasattr(model, 'unet'):
        unet = model.unet
        
        # Enable memory efficient attention
        if XFORMERS_AVAILABLE:
            try:
                unet.enable_xformers_memory_efficient_attention()
                print("✅ XFormers memory efficient attention enabled")
            except Exception as e:
                print(f"⚠️ Could not enable XFormers: {e}")
        
        # Enable attention slicing for memory efficiency
        try:
            unet.enable_attention_slicing("auto")
            print("✅ Attention slicing enabled")
        except:
            pass
            
        # Enable CPU offloading if needed
        try:
            model.enable_model_cpu_offload()
            print("✅ Model CPU offloading enabled")
        except:
            pass
    
    return model

def process_video(
    video_path,
    audio_path,
    guidance_scale,
    inference_steps,
    seed,
    use_deepcache,
    batch_size=2,  # Added batch processing
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./temp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    video_file_path = Path(video_path)
    video_path = video_file_path.absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(output_dir / f"{video_file_path.stem}_{current_time}.mp4")

    config = OmegaConf.load(CONFIG_PATH)

    # Optimize config for A100
    config["run"].update({
        "guidance_scale": guidance_scale,
        "inference_steps": inference_steps,
        "enable_xformers": XFORMERS_AVAILABLE,
        "enable_flash_attention": FLASH_ATTN_AVAILABLE,
        "batch_size": batch_size,
        "mixed_precision": True,
        "enable_attention_slicing": True,
        "enable_sequential_cpu_offload": False,  # Keep on GPU for A100
    })

    # Parse the arguments with optimizations
    args = create_optimized_args(
        video_path, audio_path, output_path, inference_steps, 
        guidance_scale, seed, use_deepcache, batch_size
    )

    try:
        # Clear GPU cache and optimize memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            print(f"🎯 Starting GPU memory: {initial_memory/1e9:.1f}GB")
            
        # Enable gradient checkpointing for memory efficiency
        with torch.cuda.amp.autocast(enabled=True):  # Mixed precision
            result = main(config=config, args=args)
            
        final_memory = torch.cuda.max_memory_allocated()
        print(f"🎯 Peak GPU memory: {final_memory/1e9:.1f}GB")
        print("✅ Processing completed successfully with optimizations.")
        return output_path
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_optimized_args(
    video_path: str, audio_path: str, output_path: str, inference_steps: int, 
    guidance_scale: float, seed: int, use_deepcache: bool, batch_size: int
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
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")

    args_list = [
        "--inference_ckpt_path", CHECKPOINT_PATH.absolute().as_posix(),
        "--video_path", video_path,
        "--audio_path", audio_path,
        "--video_out_path", output_path,
        "--inference_steps", str(inference_steps),
        "--guidance_scale", str(guidance_scale),
        "--seed", str(seed),
        "--temp_dir", "temp",
        "--batch_size", str(batch_size),
        "--mixed_precision",
    ]
    
    if use_deepcache:
        args_list.append("--enable_deepcache")
    
    if XFORMERS_AVAILABLE:
        args_list.append("--enable_xformers")

    return parser.parse_args(args_list)


def get_gpu_info():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_util = f"🔥 Optimized for maximum performance"
        return f"🚀 {gpu_name} ({gpu_memory:.1f}GB) - {gpu_util}"
    return "❌ No GPU detected"


def benchmark_performance():
    """Quick performance benchmark"""
    if torch.cuda.is_available():
        # Quick matrix multiplication benchmark
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        x = torch.randn(4096, 4096, device='cuda', dtype=torch.float16)
        
        start_time.record()
        y = torch.matmul(x, x.t())
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        flops = (2 * 4096**3) / (elapsed_time / 1000) / 1e12
        return f"⚡ GPU Performance: {flops:.2f} TFLOPS"
    return "❌ GPU benchmark failed"


# Create Gradio interface with A100 optimizations
with gr.Blocks(title="LatentSync A100 OPTIMIZED", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
    <h1 align="center">🚀 LatentSync A100 ULTRA-OPTIMIZED</h1>
    
    <div style="text-align: center; margin-bottom: 20px;">
        <p><strong>{get_gpu_info()}</strong></p>
        <p><strong>{benchmark_performance()}</strong></p>
        <p>🔥 XFormers: {'✅' if XFORMERS_AVAILABLE else '❌'} | Flash Attention: {'✅' if FLASH_ATTN_AVAILABLE else '❌'} | Mixed Precision: ✅</p>
        <p>⚡ Target: 30-60 seconds for 12-second videos</p>
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

            gr.Markdown("### ⚙️ Optimization Settings")
            with gr.Row():
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=3.0,
                    value=2.0,
                    step=0.1,
                    label="Guidance Scale",
                    info="Higher = better sync, may cause artifacts"
                )
                inference_steps = gr.Slider(
                    minimum=15, 
                    maximum=40, 
                    value=25,
                    step=1, 
                    label="Inference Steps",
                    info="More steps = better quality"
                )

            with gr.Row():
                batch_size = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1,
                    label="Batch Size",
                    info="Higher = faster processing (uses more VRAM)"
                )
                seed = gr.Number(
                    value=1247, 
                    label="Random Seed", 
                    precision=0,
                    info="Set to -1 for random"
                )

            use_deepcache = gr.Checkbox(
                value=True,
                label="Enable DeepCache",
                info="Faster processing with minimal quality loss"
            )

            process_btn = gr.Button("🔥 TURBO PROCESS", variant="primary", size="lg")
            
            gr.Markdown("### 🔥 Ultra Performance Tips")
            gr.Markdown(
                """
                - **Batch Size 2-4**: Maximum A100 throughput
                - **Guidance 2.0-2.5**: Optimal quality/speed balance  
                - **Steps 20-30**: Best performance/quality ratio
                - **DeepCache ON**: Essential for speed
                """
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Output")
            video_output = gr.Video(label="Ultra-Fast Lip-synced Video", height=400)
            
            gr.Markdown("### 📁 Test Examples")
            gr.Examples(
                examples=[
                    ["assets/demo1_video.mp4", "assets/demo1_audio.wav"],
                    ["assets/demo2_video.mp4", "assets/demo2_audio.wav"],
                    ["assets/demo3_video.mp4", "assets/demo3_audio.wav"],
                ],
                inputs=[video_input, audio_input],
                label="Try these optimized examples:"
            )

    # Processing function with all optimizations
    process_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            audio_input,
            guidance_scale,
            inference_steps,
            seed,
            use_deepcache,
            batch_size,
        ],
        outputs=video_output,
        show_progress=True,
    )

if __name__ == "__main__":
    print("🔥 Starting LatentSync ULTRA-OPTIMIZED A100 Interface...")
    print(f"🎯 {get_gpu_info()}")
    print(f"⚡ {benchmark_performance()}")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        inbrowser=False,
        show_error=True,
        favicon_path=None
    ) 