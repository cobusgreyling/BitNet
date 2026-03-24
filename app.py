"""
BitNet Explorer — Interactive visualization and benchmarking dashboard
for Microsoft's 1.58-bit LLM inference engine.

Tabs:
  1. Weight Visualizer — heatmaps of FP16 vs ternary weight distributions
  2. Benchmark Dashboard — speed, memory, energy comparisons across models/hardware
  3. Quantization Explorer — interactive slider showing compression at any bit-width
  4. API Client — connect to a running BitNet.cpp server and chat
"""

import os
import json
import math
import random
import numpy as np
import gradio as gr

# ---------------------------------------------------------------------------
# Weight Visualization — simulated weight matrices
# ---------------------------------------------------------------------------

def generate_fp16_weights(rows: int, cols: int, seed: int = 42) -> np.ndarray:
    """Generate realistic FP16 transformer weight distribution (clustered around zero)."""
    rng = np.random.RandomState(seed)
    # Mix of narrow and wider Gaussians to mimic real transformer weights
    weights = np.concatenate([
        rng.normal(0, 0.02, size=int(rows * cols * 0.7)),
        rng.normal(0, 0.08, size=int(rows * cols * 0.2)),
        rng.normal(0, 0.15, size=int(rows * cols * 0.1)),
    ])
    rng.shuffle(weights)
    return weights[:rows * cols].reshape(rows, cols).astype(np.float32)


def ternary_quantize(weights: np.ndarray) -> np.ndarray:
    """Quantize weights to {-1, 0, 1} using mean absolute value as threshold."""
    threshold = np.mean(np.abs(weights)) * 0.8
    result = np.zeros_like(weights)
    result[weights > threshold] = 1.0
    result[weights < -threshold] = -1.0
    return result


def create_weight_heatmap_data(matrix_size: int, seed: int):
    """Return FP16 and ternary weight matrices as lists for Gradio."""
    fp16 = generate_fp16_weights(matrix_size, matrix_size, seed)
    ternary = ternary_quantize(fp16)

    # Stats
    fp16_stats = (
        f"FP16 Weights ({matrix_size}x{matrix_size})\n"
        f"  Range: [{fp16.min():.4f}, {fp16.max():.4f}]\n"
        f"  Mean: {fp16.mean():.6f}\n"
        f"  Std: {fp16.std():.4f}\n"
        f"  Non-zero: {np.count_nonzero(fp16)}/{fp16.size} (100%)\n"
        f"  Memory: {fp16.size * 2:,} bytes (FP16)"
    )

    ternary_nonzero = np.count_nonzero(ternary)
    ternary_stats = (
        f"Ternary Weights ({matrix_size}x{matrix_size})\n"
        f"  Values: {{-1, 0, 1}} only\n"
        f"  -1 count: {np.sum(ternary == -1):,} ({np.sum(ternary == -1)/ternary.size*100:.1f}%)\n"
        f"   0 count: {np.sum(ternary == 0):,} ({np.sum(ternary == 0)/ternary.size*100:.1f}%)\n"
        f"  +1 count: {np.sum(ternary == 1):,} ({np.sum(ternary == 1)/ternary.size*100:.1f}%)\n"
        f"  Memory: {math.ceil(ternary.size * 1.58 / 8):,} bytes (1.58-bit)\n"
        f"  Compression: {fp16.size * 2 / max(1, math.ceil(ternary.size * 1.58 / 8)):.1f}x"
    )

    return fp16_stats, ternary_stats, fp16.tolist(), ternary.tolist()


def visualize_weights(matrix_size, seed):
    fp16_stats, ternary_stats, fp16_data, ternary_data = create_weight_heatmap_data(
        int(matrix_size), int(seed)
    )

    # Build histograms as text
    fp16_flat = np.array(fp16_data).flatten()
    ternary_flat = np.array(ternary_data).flatten()

    # FP16 histogram
    hist_counts, hist_edges = np.histogram(fp16_flat, bins=40)
    max_count = max(hist_counts)
    fp16_hist_lines = ["FP16 Weight Distribution:"]
    for i, count in enumerate(hist_counts):
        bar_len = int(count / max_count * 40) if max_count > 0 else 0
        edge = hist_edges[i]
        fp16_hist_lines.append(f"  {edge:+.3f} | {'█' * bar_len} {count}")

    # Ternary histogram
    t_minus = int(np.sum(ternary_flat == -1))
    t_zero = int(np.sum(ternary_flat == 0))
    t_plus = int(np.sum(ternary_flat == 1))
    t_max = max(t_minus, t_zero, t_plus)
    ternary_hist_lines = [
        "Ternary Weight Distribution:",
        f"  -1 | {'█' * int(t_minus / t_max * 40)} {t_minus}",
        f"   0 | {'█' * int(t_zero / t_max * 40)} {t_zero}",
        f"  +1 | {'█' * int(t_plus / t_max * 40)} {t_plus}",
    ]

    fp16_hist = "\n".join(fp16_hist_lines)
    ternary_hist = "\n".join(ternary_hist_lines)

    return fp16_stats, ternary_stats, fp16_hist, ternary_hist


# ---------------------------------------------------------------------------
# Benchmark Data
# ---------------------------------------------------------------------------

BENCHMARK_DATA = {
    "models": {
        "BitNet b1.58-2B": {"params": 2.4, "fp16_gb": 4.8, "ternary_gb": 0.5},
        "BitNet b1.58-3B": {"params": 3.3, "fp16_gb": 6.6, "ternary_gb": 0.7},
        "Llama3-8B-1.58": {"params": 8.0, "fp16_gb": 16.0, "ternary_gb": 1.6},
        "Falcon3-7B-1.58": {"params": 7.0, "fp16_gb": 14.0, "ternary_gb": 1.4},
        "BitNet b1.58-100B (projected)": {"params": 100.0, "fp16_gb": 200.0, "ternary_gb": 20.0},
    },
    "x86_speedup": {"min": 2.37, "max": 6.17},
    "arm_speedup": {"min": 1.37, "max": 5.07},
    "x86_energy_reduction": {"min": 71, "max": 82},
    "arm_energy_reduction": {"min": 55, "max": 70},
    "tokens_per_second": {
        "BitNet b1.58-2B": {"cpu": 45, "gpu_fp16": 120},
        "BitNet b1.58-3B": {"cpu": 28, "gpu_fp16": 85},
        "Llama3-8B-1.58": {"cpu": 12, "gpu_fp16": 55},
        "Falcon3-7B-1.58": {"cpu": 14, "gpu_fp16": 60},
        "BitNet b1.58-100B (projected)": {"cpu": 5, "gpu_fp16": 25},
    },
}


def generate_benchmark(model_name, hardware):
    model = BENCHMARK_DATA["models"].get(model_name)
    if not model:
        return "Select a model.", "", ""

    tps = BENCHMARK_DATA["tokens_per_second"].get(model_name, {"cpu": 10, "gpu_fp16": 50})

    if hardware == "x86 (AVX2/AVX512)":
        speedup_range = BENCHMARK_DATA["x86_speedup"]
        energy_range = BENCHMARK_DATA["x86_energy_reduction"]
    else:
        speedup_range = BENCHMARK_DATA["arm_speedup"]
        energy_range = BENCHMARK_DATA["arm_energy_reduction"]

    avg_speedup = (speedup_range["min"] + speedup_range["max"]) / 2
    avg_energy = (energy_range["min"] + energy_range["max"]) / 2

    # Memory comparison
    memory_report = (
        f"{'='*50}\n"
        f"  MEMORY COMPARISON: {model_name}\n"
        f"{'='*50}\n"
        f"\n"
        f"  Parameters:       {model['params']}B\n"
        f"  FP16 memory:      {model['fp16_gb']:.1f} GB\n"
        f"  Ternary memory:   {model['ternary_gb']:.1f} GB\n"
        f"  Compression:      {model['fp16_gb']/model['ternary_gb']:.1f}x smaller\n"
        f"  Savings:          {model['fp16_gb'] - model['ternary_gb']:.1f} GB freed\n"
        f"\n"
        f"  FP16:    {'█' * int(model['fp16_gb'] / 5)} {model['fp16_gb']:.1f} GB\n"
        f"  Ternary: {'█' * max(1, int(model['ternary_gb'] / 5))} {model['ternary_gb']:.1f} GB\n"
    )

    # Speed comparison
    speed_report = (
        f"{'='*50}\n"
        f"  SPEED & ENERGY: {hardware}\n"
        f"{'='*50}\n"
        f"\n"
        f"  Speedup range:    {speedup_range['min']}x - {speedup_range['max']}x\n"
        f"  Average speedup:  {avg_speedup:.2f}x faster\n"
        f"  Energy reduction: {energy_range['min']}% - {energy_range['max']}%\n"
        f"  Avg energy saved: {avg_energy:.0f}%\n"
        f"\n"
        f"  Tokens/sec (BitNet CPU):  {tps['cpu']}\n"
        f"  Tokens/sec (FP16 GPU):    {tps['gpu_fp16']}\n"
        f"\n"
        f"  10-step agent workflow:\n"
        f"    FP16 GPU:    ~{10 * 50 / tps['gpu_fp16']:.0f}s\n"
        f"    BitNet CPU:  ~{10 * 50 / tps['cpu']:.0f}s\n"
        f"    (assumes ~50 tokens per step)\n"
    )

    # Cost projection
    gpu_cost_hr = 3.50  # typical GPU instance
    cpu_cost_hr = 0.50  # typical CPU instance
    calls_per_hr = 3600 / (50 / tps["cpu"])  # calls per hour on CPU
    gpu_calls_per_hr = 3600 / (50 / tps["gpu_fp16"])

    cost_report = (
        f"{'='*50}\n"
        f"  COST PROJECTION (1000 agent calls/day)\n"
        f"{'='*50}\n"
        f"\n"
        f"  GPU instance (FP16):   ${gpu_cost_hr:.2f}/hr\n"
        f"  CPU instance (BitNet): ${cpu_cost_hr:.2f}/hr\n"
        f"\n"
        f"  GPU throughput:  {gpu_calls_per_hr:.0f} calls/hr\n"
        f"  CPU throughput:  {calls_per_hr:.0f} calls/hr\n"
        f"\n"
        f"  Daily cost (GPU):  ${1000 / gpu_calls_per_hr * gpu_cost_hr * 24:.2f}\n"
        f"  Daily cost (CPU):  ${1000 / calls_per_hr * cpu_cost_hr * 24:.2f}\n"
        f"  Monthly savings:   ~${(1000 / gpu_calls_per_hr * gpu_cost_hr - 1000 / calls_per_hr * cpu_cost_hr) * 24 * 30:.0f}\n"
    )

    return memory_report, speed_report, cost_report


# ---------------------------------------------------------------------------
# Quantization Explorer
# ---------------------------------------------------------------------------

def quantization_explorer(params_b, bit_width):
    """Calculate memory, compression, and ops characteristics at any bit width."""
    params = params_b * 1e9
    fp32_bytes = params * 4
    fp16_bytes = params * 2
    target_bytes = params * bit_width / 8

    fp32_gb = fp32_bytes / 1e9
    fp16_gb = fp16_bytes / 1e9
    target_gb = target_bytes / 1e9

    # Distinct values at this bit width
    distinct_values = 2 ** bit_width if bit_width >= 1 else 2

    # Relative speed (rough model: lower bits = simpler ops)
    if bit_width <= 2:
        speed_factor = "2-6x faster (addition/subtraction only)"
        ops = "No floating-point multiply. Weight × activation = add, subtract, or skip."
    elif bit_width <= 4:
        speed_factor = "1.5-3x faster (integer arithmetic)"
        ops = "Low-precision integer multiply-accumulate."
    elif bit_width <= 8:
        speed_factor = "1.2-2x faster (INT8 SIMD)"
        ops = "INT8 multiply-accumulate with SIMD acceleration."
    else:
        speed_factor = "1x baseline"
        ops = "Standard floating-point multiply-accumulate."

    report = (
        f"{'='*55}\n"
        f"  QUANTIZATION ANALYSIS: {params_b}B params @ {bit_width:.2f} bits\n"
        f"{'='*55}\n"
        f"\n"
        f"  Bit width:           {bit_width:.2f} bits per weight\n"
        f"  Distinct values:     {distinct_values:,.0f}\n"
        f"\n"
        f"  Memory footprint:\n"
        f"    FP32 (32-bit):     {fp32_gb:.1f} GB\n"
        f"    FP16 (16-bit):     {fp16_gb:.1f} GB\n"
        f"    Target ({bit_width:.1f}-bit):  {target_gb:.2f} GB\n"
        f"\n"
        f"  Compression vs FP16: {fp16_gb / target_gb:.1f}x\n"
        f"  Compression vs FP32: {fp32_gb / target_gb:.1f}x\n"
        f"  Memory saved (vs FP16): {fp16_gb - target_gb:.2f} GB ({(1 - target_gb/fp16_gb)*100:.0f}%)\n"
        f"\n"
        f"  Speed estimate:      {speed_factor}\n"
        f"  Arithmetic:          {ops}\n"
    )

    # Visual bar comparison
    max_gb = fp32_gb
    bar_scale = 50 / max_gb if max_gb > 0 else 1

    bars = (
        f"\n  Memory comparison:\n"
        f"    FP32:   {'█' * int(fp32_gb * bar_scale)} {fp32_gb:.1f} GB\n"
        f"    FP16:   {'█' * int(fp16_gb * bar_scale)} {fp16_gb:.1f} GB\n"
        f"    Target: {'█' * max(1, int(target_gb * bar_scale))} {target_gb:.2f} GB\n"
    )

    # Bit width reference table
    reference = (
        f"\n  {'='*55}\n"
        f"  BIT WIDTH REFERENCE\n"
        f"  {'='*55}\n"
        f"  32.00 bits  FP32      Full precision\n"
        f"  16.00 bits  FP16      Half precision (standard deployment)\n"
        f"   8.00 bits  INT8      Common quantization\n"
        f"   4.00 bits  INT4      Aggressive quantization (GPTQ, AWQ)\n"
        f"   2.00 bits  INT2      Extreme quantization\n"
        f"   1.58 bits  Ternary   BitNet {{-1, 0, 1}} <-- log2(3)\n"
        f"   1.00 bit   Binary    {{-1, 1}} only\n"
        f"\n"
        f"  You selected: {bit_width:.2f} bits"
    )

    return report + bars, reference


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------

def chat_with_bitnet(message, server_url, history_state):
    """Send a message to a BitNet.cpp server."""
    if not server_url.strip():
        return "Error: Enter a server URL (e.g., http://localhost:8080)", history_state, format_history(history_state)

    try:
        from openai import OpenAI
        client = OpenAI(api_key="bitnet-local", base_url=server_url.rstrip("/") + "/v1")

        history = history_state or []
        history.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model="bitnet",
            messages=history,
            max_tokens=512,
            temperature=0.7,
        )

        assistant_msg = response.choices[0].message.content
        history.append({"role": "assistant", "content": assistant_msg})

        # Build stats
        usage = response.usage
        stats = ""
        if usage:
            stats = (
                f"Tokens: {usage.prompt_tokens} in + {usage.completion_tokens} out = {usage.total_tokens}\n"
                f"Model: {response.model}"
            )

        return format_history(history), history, stats

    except ImportError:
        return "Error: openai package not installed. Run: pip install openai", history_state, ""
    except Exception as exc:
        return f"Error: {exc}\n\nMake sure BitNet.cpp server is running at {server_url}", history_state, ""


def format_history(history):
    if not history:
        return "No messages yet. Send a message to start chatting."
    lines = []
    for msg in history:
        role = msg["role"].upper()
        lines.append(f"[{role}] {msg['content']}")
        lines.append("")
    return "\n".join(lines)


def clear_history():
    return [], "History cleared."


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CSS = """
.mono textarea { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace !important; font-size: 13px !important; }
.header-text { font-size: 15px !important; }
"""

THEME = gr.themes.Base(
    primary_hue="green",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#0f172a",
    body_background_fill_dark="#0f172a",
    block_background_fill="#1e293b",
    block_background_fill_dark="#1e293b",
    block_border_color="#334155",
    block_border_color_dark="#334155",
    block_label_text_color="#94a3b8",
    block_label_text_color_dark="#94a3b8",
    block_title_text_color="#f8fafc",
    block_title_text_color_dark="#f8fafc",
    body_text_color="#e2e8f0",
    body_text_color_dark="#e2e8f0",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    button_primary_background_fill="#10b981",
    button_primary_background_fill_dark="#10b981",
    button_primary_background_fill_hover="#34d399",
    button_primary_background_fill_hover_dark="#34d399",
    button_primary_text_color="#0f172a",
    button_primary_text_color_dark="#0f172a",
    input_background_fill="#1e293b",
    input_background_fill_dark="#1e293b",
    input_border_color="#475569",
    input_border_color_dark="#475569",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
    border_color_primary="#10b981",
    border_color_primary_dark="#10b981",
)

with gr.Blocks(title="BitNet Explorer", theme=THEME, css=CSS) as demo:

    gr.Markdown("# BitNet Explorer")
    gr.Markdown(
        "Interactive visualization and benchmarking dashboard for Microsoft's **BitNet** — "
        "1.58-bit LLM inference where every weight is just **-1, 0, or 1**."
    )

    with gr.Tabs():

        # ============================================================
        # TAB 1: Weight Visualizer
        # ============================================================
        with gr.Tab("Weight Visualizer"):
            gr.Markdown(
                "### FP16 vs Ternary Weight Distribution\n"
                "Visualize how transformer weights look before and after ternary quantization. "
                "Real transformer weights cluster around zero — BitNet exploits this by mapping "
                "to just three values: {-1, 0, 1}."
            )

            with gr.Row():
                viz_size = gr.Slider(minimum=16, maximum=128, value=64, step=16, label="Matrix Size")
                viz_seed = gr.Slider(minimum=1, maximum=100, value=42, step=1, label="Random Seed")
                viz_btn = gr.Button("Generate Weights", variant="primary")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### FP16 (Full Precision)")
                    fp16_stats = gr.Textbox(label="Statistics", lines=7, interactive=False, elem_classes=["mono"])
                    fp16_hist = gr.Textbox(label="Distribution", lines=15, interactive=False, elem_classes=["mono"])
                with gr.Column():
                    gr.Markdown("### Ternary (1.58-bit)")
                    ternary_stats = gr.Textbox(label="Statistics", lines=7, interactive=False, elem_classes=["mono"])
                    ternary_hist = gr.Textbox(label="Distribution", lines=15, interactive=False, elem_classes=["mono"])

            viz_btn.click(
                fn=visualize_weights,
                inputs=[viz_size, viz_seed],
                outputs=[fp16_stats, ternary_stats, fp16_hist, ternary_hist],
            )

        # ============================================================
        # TAB 2: Benchmark Dashboard
        # ============================================================
        with gr.Tab("Benchmarks"):
            gr.Markdown(
                "### Speed, Memory & Cost Benchmarks\n"
                "Select a model and hardware platform to see projected performance gains, "
                "memory savings, and cost comparisons for AI agent workloads."
            )

            with gr.Row():
                bench_model = gr.Dropdown(
                    choices=list(BENCHMARK_DATA["models"].keys()),
                    value="BitNet b1.58-3B",
                    label="Model",
                )
                bench_hw = gr.Dropdown(
                    choices=["x86 (AVX2/AVX512)", "ARM (NEON)"],
                    value="x86 (AVX2/AVX512)",
                    label="Hardware",
                )
                bench_btn = gr.Button("Run Benchmark", variant="primary")

            with gr.Row():
                bench_memory = gr.Textbox(label="Memory", lines=12, interactive=False, elem_classes=["mono"])
                bench_speed = gr.Textbox(label="Speed & Energy", lines=14, interactive=False, elem_classes=["mono"])
                bench_cost = gr.Textbox(label="Cost Projection", lines=14, interactive=False, elem_classes=["mono"])

            bench_btn.click(
                fn=generate_benchmark,
                inputs=[bench_model, bench_hw],
                outputs=[bench_memory, bench_speed, bench_cost],
            )

        # ============================================================
        # TAB 3: Quantization Explorer
        # ============================================================
        with gr.Tab("Quantization Explorer"):
            gr.Markdown(
                "### Explore Any Bit Width\n"
                "Slide from 1-bit to 32-bit and see how memory, compression, and arithmetic change. "
                "BitNet operates at **1.58 bits** — the sweet spot where ternary values capture "
                "the essential structure of transformer weights."
            )

            with gr.Row():
                quant_params = gr.Slider(minimum=0.5, maximum=200, value=3.3, step=0.1, label="Parameters (Billions)")
                quant_bits = gr.Slider(minimum=1.0, maximum=32.0, value=1.58, step=0.01, label="Bits per Weight")
                quant_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                quant_report = gr.Textbox(label="Analysis", lines=20, interactive=False, elem_classes=["mono"])
                quant_reference = gr.Textbox(label="Reference", lines=16, interactive=False, elem_classes=["mono"])

            quant_btn.click(
                fn=quantization_explorer,
                inputs=[quant_params, quant_bits],
                outputs=[quant_report, quant_reference],
            )

        # ============================================================
        # TAB 4: API Client
        # ============================================================
        with gr.Tab("API Client"):
            gr.Markdown(
                "### Chat with a BitNet.cpp Server\n"
                "Connect to a running BitNet.cpp server's OpenAI-compatible API. "
                "Start a server with:\n"
                "```\n"
                "python run_inference_server.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf --port 8080\n"
                "```"
            )

            history_state = gr.State([])

            with gr.Row():
                server_url = gr.Textbox(
                    label="Server URL",
                    value="http://localhost:8080",
                    scale=3,
                )
                clear_btn = gr.Button("Clear History", scale=1)

            chat_display = gr.Textbox(
                label="Conversation",
                lines=20,
                interactive=False,
                elem_classes=["mono"],
                value="No messages yet. Send a message to start chatting.",
            )

            with gr.Row():
                chat_input = gr.Textbox(
                    label="Message",
                    placeholder="Ask something...",
                    scale=4,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)

            chat_stats = gr.Textbox(label="Response Stats", lines=2, interactive=False, elem_classes=["mono"])

            send_btn.click(
                fn=chat_with_bitnet,
                inputs=[chat_input, server_url, history_state],
                outputs=[chat_display, history_state, chat_stats],
            )
            chat_input.submit(
                fn=chat_with_bitnet,
                inputs=[chat_input, server_url, history_state],
                outputs=[chat_display, history_state, chat_stats],
            )
            clear_btn.click(
                fn=lambda: ([], "History cleared.", ""),
                outputs=[history_state, chat_display, chat_stats],
            )

        # ============================================================
        # TAB 5: Model Reference
        # ============================================================
        with gr.Tab("Models & Setup"):
            gr.Markdown(
                "### Supported Models\n\n"
                "| Model | Parameters | FP16 Size | BitNet Size | Compression |\n"
                "|-------|-----------|-----------|-------------|-------------|\n"
                "| BitNet b1.58-2B-4T | 2.4B | 4.8 GB | 0.5 GB | 9.6x |\n"
                "| BitNet b1.58-3B | 3.3B | 6.6 GB | 0.7 GB | 9.4x |\n"
                "| BitNet b1.58-Large | 0.7B | 1.4 GB | 0.15 GB | 9.3x |\n"
                "| Llama3-8B-1.58 | 8B | 16 GB | 1.6 GB | 10x |\n"
                "| Falcon3-1B-1.58 | 1B | 2 GB | 0.2 GB | 10x |\n"
                "| Falcon3-3B-1.58 | 3B | 6 GB | 0.6 GB | 10x |\n"
                "| Falcon3-7B-1.58 | 7B | 14 GB | 1.4 GB | 10x |\n"
                "| Falcon3-10B-1.58 | 10B | 20 GB | 2 GB | 10x |\n"
                "\n---\n"
                "### Quick Start\n"
                "```bash\n"
                "# Clone BitNet.cpp\n"
                "git clone https://github.com/microsoft/BitNet.git\n"
                "cd BitNet\n"
                "\n"
                "# Download and compile a 3B model\n"
                "python setup_env.py --hf-repo microsoft/BitNet-b1.58-3B --quant-type i2_s\n"
                "\n"
                "# Run inference\n"
                "python run_inference.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf \\\n"
                "  -p \"Explain quantum computing\" -t 8\n"
                "\n"
                "# Start API server\n"
                "python run_inference_server.py -m models/bitnet_b1_58-3B/ggml-model-i2_s.gguf \\\n"
                "  --port 8080 --threads 8\n"
                "```\n"
                "\n---\n"
                "### Or use the setup script from this repo\n"
                "```bash\n"
                "chmod +x setup-bitnet.sh\n"
                "./setup-bitnet.sh\n"
                "```\n"
                "The script clones BitNet.cpp, downloads a model, compiles, and starts the server."
            )

    gr.Markdown("---\n*BitNet Explorer — Cobus Greyling*")

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
