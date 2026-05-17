from transformers import CLIPModel

model_id = "qihoo360/fg-clip-base"
print(f"Loading {model_id}...")

base_model = CLIPModel.from_pretrained(model_id)

total = sum(p.numel() for p in base_model.parameters())
trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

print(f"\nTotal parameters: {total:,}")
print(f"Trainable parameters: {trainable:,}")
print(f"Frozen parameters: {total - trainable:,}")

# Break down by component
text_params = sum(p.numel() for p in base_model.text_model.parameters())
vision_params = sum(p.numel() for p in base_model.vision_model.parameters())
logit_scale = base_model.logit_scale.exp().item()
print(f"\nText encoder params: {text_params:,}")
print(f"Vision encoder params: {vision_params:,}")
print(f"Text + Vision + logit_scale: {text_params + vision_params + 1:,}")
print(f"logit_scale value: {logit_scale:.4f} (temp = {1/logit_scale:.6f})")