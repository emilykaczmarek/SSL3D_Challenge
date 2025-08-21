import torch

checkpoint = torch.load("/home/e/ekacz/links/scratch/nnssl/nnssl_results/Dataset745_OpenMind/SimCLRTrainer__nnsslPlans__onemmiso/fold_all/checkpoint_latest.pth", map_location="cpu")

# List top-level keys
print(checkpoint.keys())
logs = checkpoint.get("logging", {})
for k, v in logs.items():
    print(f"{k}: {len(v)} entries")
