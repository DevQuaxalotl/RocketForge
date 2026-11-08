"""Rebuild Necto and Nexto as device-agnostic TorchScript policies."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class PerceiverBlock(nn.Module):
    def __init__(self, feedforward_size: int, use_layer_norm: bool):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.linear1 = nn.Linear(128, feedforward_size)
        self.linear2 = nn.Linear(feedforward_size, 128)
        self.norm1 = nn.LayerNorm(128) if use_layer_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(128) if use_layer_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(128) if use_layer_norm else nn.Identity()

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        attention, _ = self.attention(
            self.norm1(query),
            self.norm2(key_value),
            self.norm2(key_value),
            key_padding_mask=key_padding_mask.to(torch.bool),
            need_weights=False,
        )
        output = query + attention
        hidden = torch.relu(self.linear1(self.norm3(output)))
        return output + self.linear2(hidden)


class NectoGpuPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_preprocess = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.key_value_preprocess = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.block = PerceiverBlock(256, False)
        self.output = nn.Linear(128, 12)

    def forward(self, obs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        query, key_value, mask = obs
        query = self.query_preprocess(query)
        key_value = self.key_value_preprocess(key_value)
        query = self.block(query, key_value, mask)
        heads = torch.split(self.output(torch.relu(query)), [3, 3, 2, 2, 2], -1)
        joint = heads[0].squeeze(1)
        for head in heads[1:]:
            joint = (joint.unsqueeze(-1) + head.squeeze(1).unsqueeze(-2)).flatten(1)
        return joint


class NextoGpuPolicy(nn.Module):
    def __init__(self, lookup_table: Tensor):
        super().__init__()
        self.query_preprocess = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.key_value_preprocess = nn.Sequential(
            nn.Linear(24, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [PerceiverBlock(512, True), PerceiverBlock(512, True)]
        )
        self.action_net = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.embedding = nn.Linear(128, 32)
        self.register_buffer("lookup_table", lookup_table)

    def forward(self, obs: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        query, key_value, mask = obs
        query = self.query_preprocess(query)
        key_value = self.key_value_preprocess(key_value)
        for block in self.blocks:
            query = block(query, key_value, mask)
        query_embedding = self.embedding(torch.relu(query))
        action_embedding = self.action_net(self.lookup_table)
        return torch.einsum("ad,bpd->bpa", action_embedding, query_embedding).squeeze(1)


def make_nexto_lookup_table() -> Tensor:
    actions = []
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle or boost, steer, 0, steer, 0, 0, boost, handbrake]
                    )
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (
                            pitch != 0 or yaw != 0 or roll != 0
                        )
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                        )
    result = torch.tensor(actions, dtype=torch.float32)
    if result.shape != (90, 8):
        raise RuntimeError(f"unexpected Nexto lookup shape: {tuple(result.shape)}")
    return result


def copy_tensor(target: Tensor, source: Tensor) -> None:
    with torch.no_grad():
        target.copy_(source)


def copy_linear(target: nn.Linear, state: Dict[str, Tensor], prefix: str) -> None:
    copy_tensor(target.weight, state[f"{prefix}.weight"])
    copy_tensor(target.bias, state[f"{prefix}.bias"])


def copy_preprocess(
    target: nn.Sequential, state: Dict[str, Tensor], prefix: str
) -> None:
    copy_linear(target[0], state, f"{prefix}.0")
    copy_linear(target[2], state, f"{prefix}.2")


def copy_block(
    target: PerceiverBlock,
    state: Dict[str, Tensor],
    prefix: str,
) -> None:
    copy_tensor(
        target.attention.in_proj_weight, state[f"{prefix}.attention.in_proj_weight"]
    )
    copy_tensor(
        target.attention.in_proj_bias, state[f"{prefix}.attention.in_proj_bias"]
    )
    copy_linear(target.attention.out_proj, state, f"{prefix}.attention.out_proj")
    copy_linear(target.linear1, state, f"{prefix}.linear1")
    copy_linear(target.linear2, state, f"{prefix}.linear2")
    if target.use_layer_norm:
        for name in ("norm1", "norm2", "norm3"):
            norm = getattr(target, name)
            copy_tensor(norm.weight, state[f"{prefix}.{name}.weight"])
            copy_tensor(norm.bias, state[f"{prefix}.{name}.bias"])


def load_necto(source_path: Path) -> Tuple[torch.jit.ScriptModule, NectoGpuPolicy]:
    original = torch.jit.load(str(source_path), map_location="cpu").eval()
    state = original.state_dict()
    converted = NectoGpuPolicy().eval()
    copy_preprocess(
        converted.query_preprocess, state, "net.earl.query_preprocess"
    )
    copy_preprocess(
        converted.key_value_preprocess, state, "net.earl.key_value_preprocess"
    )
    copy_block(converted.block, state, "net.earl.blocks.0")
    copy_linear(converted.output, state, "net.output.linear")
    return original, converted


def load_nexto(source_path: Path) -> Tuple[torch.jit.ScriptModule, NextoGpuPolicy]:
    original = torch.jit.load(str(source_path), map_location="cpu").eval()
    state = original.state_dict()
    converted = NextoGpuPolicy(make_nexto_lookup_table()).eval()
    copy_preprocess(
        converted.query_preprocess, state, "net.earl.query_preprocess"
    )
    copy_preprocess(
        converted.key_value_preprocess, state, "net.earl.key_value_preprocess"
    )
    for index, block in enumerate(converted.blocks):
        copy_block(block, state, f"net.earl.blocks.{index}")
    copy_preprocess(converted.action_net, state, "net.output.net")
    copy_linear(converted.action_net[4], state, "net.output.net.4")
    copy_linear(converted.embedding, state, "net.output.emb_convertor")
    return original, converted


def original_necto_logits(output) -> Tensor:
    heads = output[0]
    joint = heads[0].squeeze(1)
    for head in heads[1:]:
        joint = (joint.unsqueeze(-1) + head.squeeze(1).unsqueeze(-2)).flatten(1)
    return joint


def original_nexto_logits(output) -> Tensor:
    return output[0].squeeze(1)


def verify(
    name: str,
    original: torch.jit.ScriptModule,
    converted: nn.Module,
    original_logits,
) -> None:
    generator = torch.Generator().manual_seed(0x4E455854)
    worst_error = 0.0
    frames = 0
    with torch.no_grad():
        for batch_size in (1, 7, 64):
            for iteration in range(4):
                query = torch.randn(batch_size, 1, 32, generator=generator)
                key_value = torch.randn(batch_size, 41, 24, generator=generator)
                mask = torch.zeros(batch_size, 41)
                inputs = (query, key_value, mask)
                expected = original_logits(original(inputs))
                actual = converted(inputs)
                error = (expected - actual).abs().max().item()
                worst_error = max(worst_error, error)
                if not torch.equal(expected.argmax(1), actual.argmax(1)):
                    raise RuntimeError(
                        f"{name} conversion changed deterministic actions "
                        f"for batch {batch_size}, iteration {iteration}"
                    )
                frames += batch_size

        # The published wrappers use variable entity counts. RocketForge pads
        # to the 3v3 width, so masked padding must be numerically equivalent.
        for player_count in (2, 4, 6):
            query = torch.randn(32, 1, 32, generator=generator)
            key_value = torch.randn(
                32,
                player_count + 35,
                24,
                generator=generator,
            )
            variable_mask = torch.zeros(32, player_count + 35, dtype=torch.bool)
            expected = original_logits(original((query, key_value, variable_mask)))

            fixed_key_value = torch.zeros(32, 41, 24)
            fixed_mask = torch.zeros(32, 41)
            if name == "Necto":
                fixed_key_value[:, 0] = key_value[:, 0]
                fixed_key_value[:, 1:1 + player_count] = key_value[
                    :, 1:1 + player_count
                ]
                fixed_key_value[:, 7:] = key_value[:, 1 + player_count:]
                fixed_mask[:, 1 + player_count:7] = 1
            else:
                fixed_key_value[:, :player_count] = key_value[:, :player_count]
                fixed_key_value[:, 6] = key_value[:, player_count]
                fixed_key_value[:, 7:] = key_value[:, player_count + 1:]
                fixed_mask[:, player_count:6] = 1

            actual = converted((query, fixed_key_value, fixed_mask))
            error = (expected - actual).abs().max().item()
            worst_error = max(worst_error, error)
            if not torch.equal(expected.argmax(1), actual.argmax(1)):
                raise RuntimeError(
                    f"{name} padded {player_count}-player inputs changed actions"
                )
            frames += query.size(0)
    if worst_error > 5e-4:
        raise RuntimeError(
            f"{name} conversion exceeded logit tolerance: {worst_error:.9g}"
        )
    print(
        f"{name}: {frames} randomized frames, identical actions, "
        f"max |logit error|={worst_error:.9g}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--necto-source", type=Path, required=True)
    parser.add_argument("--nexto-source", type=Path, required=True)
    parser.add_argument("--necto-output", type=Path, required=True)
    parser.add_argument("--nexto-output", type=Path, required=True)
    args = parser.parse_args()

    original_necto, converted_necto = load_necto(args.necto_source)
    original_nexto, converted_nexto = load_nexto(args.nexto_source)
    verify("Necto", original_necto, converted_necto, original_necto_logits)
    verify("Nexto", original_nexto, converted_nexto, original_nexto_logits)

    args.necto_output.parent.mkdir(parents=True, exist_ok=True)
    args.nexto_output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.script(converted_necto).save(str(args.necto_output))
    torch.jit.script(converted_nexto).save(str(args.nexto_output))
    print(f"Necto GPU-safe policy: {args.necto_output}")
    print(f"Nexto GPU-safe policy: {args.nexto_output}")


if __name__ == "__main__":
    main()
