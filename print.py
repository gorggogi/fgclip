from safetensors.torch import load_file
import os, re

base = r"C:\Users\Charlene C. Dilig\OneDrive\Documents\Github\fgclip\lorafinetuned"
models = {
    "V1": "fgclip-lora-finetunedV1-1200",
    "V2": "fgclip-lora-finetunedV2-1200-HNM",
    "V3": "fgclip-lora-finetunedV3-optimized",
}

# Load the base model directly from HuggingFace
from transformers import CLIPModel
base_clip = CLIPModel.from_pretrained("qihoo360/fg-clip-base")
base_params = sum(p.numel() for p in base_clip.parameters())
print(f"=== qihoo360/fg-clip-base (loaded directly) ===")
print(f"  Total parameters: {base_params:,}")
print()

for name, model_dir in models.items():
    sf_path = os.path.join(base, model_dir, "adapter_model.safetensors")
    sf = load_file(sf_path)
    lora_params = sum(t.numel() for t in sf.values())

    # Find total params in training_summary_report.txt via regex
    cfg_path = os.path.join(base, model_dir, "training_summary_report.txt")
    total_reported = None
    with open(cfg_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.search(r"Trainable Params:\s*([\d,]+)\s*\(([\d.]+)%\s*of\s*total\)", line)
            if m:
                total_str = m.group(1).replace(",", "")
                total_reported = int(total_str)
                break

    if total_reported:
        base_model = total_reported - lora_params
        match = "MATCH" if base_model == base_params else "MISMATCH"
        print(f"=== {name} ===")
        print(f"  LoRA params (safetensors):  {lora_params:,}")
        print(f"  Total reported (notebook):  {total_reported:,}")
        print(f"  Base model (computed):      {base_model:,}")
        print(f"  Direct base check:          {base_params:,} ({match})")
        print()
    else:
        print(f"=== {name} === (could not parse total params)")
        print()