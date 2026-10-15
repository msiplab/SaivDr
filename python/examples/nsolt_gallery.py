"""Atomic images of NSOLTs with random parameters (cf. examples/dcnn/main_nsoltnw_gallery.m).

python nsolt_gallery.py [--output gallery.png]
"""

import argparse
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from saivdr.dcnn import atomic_imshow, create_nsolt_2d  # noqa: E402

CONFIGS = [
    dict(number_of_channels=(2, 2), decimation_factor=(2, 2), polyphase_order=(0, 0)),
    dict(number_of_channels=(3, 3), decimation_factor=(2, 2), polyphase_order=(2, 2)),
    dict(number_of_channels=(5, 5), decimation_factor=(2, 2), polyphase_order=(4, 4)),
    dict(number_of_channels=(4, 4), decimation_factor=(2, 2), polyphase_order=(2, 2), number_of_levels=2),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="nsolt_gallery.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(4 * len(CONFIGS), 4))
    for ax, cfg in zip(axes, CONFIGS):
        syn = create_nsolt_2d(mode="Synthesizer", **cfg)
        with torch.no_grad():
            for p in syn.parameters():
                p.normal_(0.0, math.pi / 6)
        atomic_imshow(syn, ax=ax)
        ax.set_title(f"P={sum(cfg['number_of_channels'])}, N={cfg['polyphase_order']}")
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
