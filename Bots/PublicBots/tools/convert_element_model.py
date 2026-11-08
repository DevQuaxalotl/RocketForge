"""Convert RLBotPack Element's public pickle weights to a TorchScript policy."""

import argparse
import pickle
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-path", required=True)
    parser.add_argument("--botpack-root", required=True)
    parser.add_argument("--element-output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.torch_path)

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ElementActor(nn.Module):
        def __init__(self, input_size: int = 107):
            super().__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 256)
            self.fc5 = nn.Linear(256, 256)
            self.cat_heads = nn.Linear(256, 15)
            self.ber_heads = nn.Linear(256, 6)

        def features(self, x: torch.Tensor) -> torch.Tensor:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return F.relu(self.fc5(x))

        def forward(self, x: torch.Tensor):
            x = self.features(x)
            categorical = self.cat_heads(x).view(-1, 5, 3)
            bernoulli = self.ber_heads(x).view(-1, 3, 2)
            return categorical, bernoulli

    class ElementJointPolicy(nn.Module):
        def __init__(self, actor: ElementActor):
            super().__init__()
            self.actor = actor

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            categorical, bernoulli = self.actor(obs)
            joint = categorical[:, 0, :]
            for head_index in range(1, 5):
                head = categorical[:, head_index, :]
                joint = (joint.unsqueeze(-1) + head.unsqueeze(1)).flatten(1)
            for head_index in range(3):
                head = bernoulli[:, head_index, :]
                joint = (joint.unsqueeze(-1) + head.unsqueeze(1)).flatten(1)
            return joint

    botpack_root = Path(args.botpack_root)
    element_pickle = botpack_root / "RLBotPack" / "Element" / "model.p"
    with element_pickle.open("rb") as stream:
        state_dict = pickle.load(stream)

    actor = ElementActor()
    actor.load_state_dict(state_dict)
    actor.eval()

    policy = ElementJointPolicy(actor).eval()
    scripted = torch.jit.script(policy)
    output_path = Path(args.element_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output_path))

    # Verify that joint-logit argmax decodes to the original per-head argmax.
    generator = torch.Generator().manual_seed(0x454C454D)
    max_logit_error = 0.0
    verified_frames = 0
    with torch.no_grad():
        for batch_size in (1, 7, 64):
            for iteration in range(4):
                sample = torch.randn(
                    batch_size,
                    107,
                    generator=generator,
                )
                categorical, bernoulli = actor(sample)
                expected = torch.cat(
                    [categorical.argmax(dim=2), bernoulli.argmax(dim=2)],
                    dim=1,
                )
                eager_logits = policy(sample)
                scripted_logits = scripted(sample)
                max_logit_error = max(
                    max_logit_error,
                    (eager_logits - scripted_logits).abs().max().item(),
                )
                joint_index = scripted_logits.argmax(dim=1)

                decoded = []
                remaining = joint_index.clone()
                radices = [3, 3, 3, 3, 3, 2, 2, 2]
                for radix in reversed(radices):
                    decoded.append(remaining.remainder(radix))
                    remaining = torch.div(
                        remaining,
                        radix,
                        rounding_mode="floor",
                    )
                actual = torch.stack(list(reversed(decoded)), dim=1)
                if not torch.equal(actual, expected):
                    raise RuntimeError(
                        "Element joint-policy conversion changed deterministic "
                        f"actions for batch {batch_size}, iteration {iteration}"
                    )
                verified_frames += batch_size

    print(f"Element scripted policy: {output_path} ({output_path.stat().st_size} bytes)")
    print(f"Element output shape: {tuple(scripted(torch.zeros(1, 107)).shape)}")
    print(
        f"Element: {verified_frames} randomized frames, identical actions, "
        f"max |logit error|={max_logit_error:.9g}"
    )


if __name__ == "__main__":
    main()
