#!/usr/bin/env python3
"""
Analysis for MicroICS inference output.

Using the feature values of treated (unknown) images, the model’s predictions,
and the training datafile with feature values from the training set, this script helps you visualise how each treatment shifts a strain’s structure across the top features – and towards which class.

===========================================================================
IMAGE NAMING / HOW THE DATA IS PARSED – CRUCIAL FOR PROPER WORKING OF THE SCRIPT
===========================================================================
The script reads the STRAIN and the TREATMENT of each image directly from the
image name (the index of the features/predictions tables). Please check the description of the naming convention in the image preparation requirements. It uses the
MicroICS naming convention used in the datafiles, where fields are separated by double hyphens “--”
and each field is a “--key--value” pair. Example image name:

    05--03--2024--s--Lm--st--L628--p--B02--pos003--tm--24--ch--Syto9--z--21--treat--milkcoat
                              ^^^^                                              ^^^^^^^^^
                              strain (after --st--)                            treatment (after --treat--)

  * STRAIN    is taken from the “--st--” field    -> here: L628
  * TREATMENT is taken from the “--treat--” field -> here: milkcoat

The parsed STRAIN must match the entries in the training datafile’s “label”
column, because the origin/destination reference bands are looked up there.
For the identity task the strain name IS the class label (L628, L1764, …).

TRUE CLASS: the value after “--st--” IS the class label – a strain name for the
identity task, or a relabelled code if you have changed the
label. So the true class of every image is simply its “--st--” value; nothing
extra needs to be supplied, and a run containing several classes is handled
correctly (each is judged against its own “--st--” label).

If your file names use different markers, override the parsing with
--strain-regex / --treat-regex (each a regex with one capture group). The
defaults tolerate hyphens and spaces inside a name (they stop at the “--”
field separator), e.g. “ATCC 19115” or “L1764-GFP” are captured whole.

===========================================================================
VIEWS
===========================================================================
  1. BAND view (--mode band):
     control -> treatments as points/medians, with shaded bands showing the
     TRAINING range of reference classes: BOTH the strain’s own class (origin,
     blue) and a destination class (gold). You see the sample move away from
     its origin band towards the destination band, feature by feature.
     Set them with --origin / --destination (either may be omitted; origin
     defaults to the strain’s own class, destination to the most common
     predicted class among the treated images).

  2. PREDICTED-CLASS view (--mode predclass):
     every treated image is drawn with a COLOUR and a distinct MARKER SYMBOL
     for the class the model PREDICTED for it, with horizontal jitter so
     overlapping classes are visible. This shows which class each image becomes and
     whether converted images cluster coherently (real conversion) or scatter
     (forced calls).

Both modes print a MAJORITY-CLASS table: per strain × treatment, the most
common predicted class and its fraction, flagged against that strain’s own
true class.

===========================================================================
INPUTS
===========================================================================
  --features    rf_features.tsv feature VALUES of the treated images
 (index = image name, columns = features)
  --predictions rf_predictions.tsv predicted class per image
 (index = image name, “prediction” column)
  --datafile    datafile.tsv TRAINING data; “label” column = class,
 used for the per-class reference bands
  --importance  rf_feature_importance.csv feature ranking (cols: feature, importance)

===========================================================================
EXAMPLES
===========================================================================
Only the four input files and --mode are required. Everything else has a
sensible default: without --strain, ALL strains are processed (one figure
each); the destination band is auto-chosen; the origin band is the strain’s
own class.

  # minimal: process every strain, auto origin + destination bands
  python treatment_shift.py --features rf_features.tsv --predictions rf_predictions.tsv --datafile datafile.tsv --importance rf_feature_importance.csv --mode band --out-dir ./treat_out

  # per-treatment destination bands (each treatment vs its own majority class)
  python treatment_shift.py --features rf_features.tsv --predictions rf_predictions.tsv --datafile datafile.tsv --importance rf_feature_importance.csv --mode perdest --out-dir ./treat_out

  # predicted-class view (colour + symbol + jitter), every strain
  python treatment_shift.py --features rf_features.tsv --predictions rf_predictions.tsv --datafile datafile.tsv --importance rf_feature_importance.csv --mode predclass --out-dir ./treat_out

  # OPTIONAL refinements:
  #   --strain L628         look at one strain only (else all strains)
  #   --destination L1764   force the destination band (else auto = majority class)

"""

import argparse
import math
import os
import re
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "p", "h"]


def parse_meta(names, strain_rx, treat_rx):
    srx, trx = re.compile(strain_rx), re.compile(treat_rx)
    strains, treats = [], []
    for n in names:
        s = srx.search(str(n)); t = trx.search(str(n))
        strains.append(s.group(1).strip() if s else "?")
        treats.append(t.group(1).strip() if t else "?")
    return strains, treats


def order_treatments(found, preferred):
    found = list(found)
    ordered = [t for t in preferred if t in found]
    ordered += [t for t in found if t not in ordered]
    return ordered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--datafile", required=True)
    ap.add_argument("--importance", required=True)
    ap.add_argument("--mode", choices=["band", "predclass", "perdest"], default="band")
    ap.add_argument("--n", type=int, default=20, help="top-N features to plot")
    ap.add_argument("--destination", default=None,
                    help="(band mode) training class shown as the destination band; "
                         "default: the most common predicted class among treated images")
    ap.add_argument("--origin", default=None,
                    help="(band mode) training class shown as the origin band; "
                         "default: the strain's own class")
    ap.add_argument("--strain", default=None,
                    help="restrict to one strain (default: all strains, one figure each)")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--pred-col", default="prediction")
    # robust defaults: capture up to the "--" field separator, so hyphens/spaces
    # inside a name are kept whole
    ap.add_argument("--strain-regex", default=r"--st--(.+?)--")
    ap.add_argument("--treat-regex", default=r"--treat--(.+?)(?:--|$)")
    ap.add_argument("--treat-order", default="CTRL,meloncoat,milkcoat,tatarsteakcoat",
                    help="comma-separated preferred treatment order (control first)")
    ap.add_argument("--out-dir", default="./treat_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rf = pd.read_csv(args.features, sep="\t", index_col=0)
    pred = pd.read_csv(args.predictions, sep="\t", index_col=0)[args.pred_col]
    train = pd.read_csv(args.datafile, sep="\t")
    imp = pd.read_csv(args.importance).sort_values("importance", ascending=False)

    strains, treats = parse_meta(rf.index, args.strain_regex, args.treat_regex)
    meta = pd.DataFrame({"strain": strains, "treat": treats}, index=rf.index)
    meta["pred"] = pred.reindex(rf.index).values

    # the value after --st-- IS the class label (strain name, CN, CI, ...),
    # so the true class of every image is simply its parsed --st-- value.
    meta["true"] = meta["strain"]

    top = [f for f in imp["feature"] if f in rf.columns and f in train.columns][:args.n]
    if not top:
        print("ERROR: no top features found in both features and datafile.", file=sys.stderr)
        sys.exit(1)

    pref = [t.strip() for t in args.treat_order.split(",") if t.strip()]
    treat_list = order_treatments(sorted(meta["treat"].unique()), pref)
    short_treat = {t: (t[:4]) for t in treat_list}
    if pref:
        short_treat[pref[0]] = "CTRL"

    strains_to_do = [args.strain] if args.strain else sorted(meta["strain"].unique())

    all_classes = sorted(set(meta["pred"].unique()) | set(train[args.label_col].unique()))
    cmap = plt.get_cmap("tab10")
    class_colour = {c: cmap(i % 10) for i, c in enumerate(all_classes)}
    class_marker = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(all_classes)}

    ncol = 5
    nrow = math.ceil(len(top) / ncol)

    for strain in strains_to_do:
        sm = meta[meta["strain"] == strain]
        if len(sm) == 0:
            print(f"NOTE: no images for strain {strain}", file=sys.stderr); continue

        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow))
        axes = np.array(axes).flatten()

        if args.mode == "band":
            dest = args.destination or sm["pred"].value_counts().idxmax()
            origin = args.origin
            if origin is None and strain in set(train[args.label_col].unique()):
                origin = strain
            dest_vals = train[train[args.label_col] == dest]
            origin_vals = train[train[args.label_col] == origin] if origin else None

            palette = ["#333333", "#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
            tcol = {t: palette[j % len(palette)] for j, t in enumerate(treat_list)}

            for k, feat in enumerate(top):
                ax = axes[k]
                ref = dest_vals[feat].astype(float)
                ax.axhspan(ref.quantile(0.25), ref.quantile(0.75), color="gold", alpha=0.18)
                ax.axhline(ref.median(), color="goldenrod", lw=1, ls="--")
                if origin_vals is not None:
                    o = origin_vals[feat].astype(float)
                    ax.axhspan(o.quantile(0.25), o.quantile(0.75), color="steelblue", alpha=0.14)
                    ax.axhline(o.median(), color="steelblue", lw=1, ls=":")
                for j, t in enumerate(treat_list):
                    ids = sm[sm["treat"] == t].index
                    v = rf.loc[ids, feat].astype(float)
                    ax.scatter([j] * len(v), v, alpha=0.4, s=12, color=tcol.get(t, "gray"))
                    if len(v):
                        ax.scatter([j], [v.median()], marker="_", s=400,
                                   color=tcol.get(t, "gray"), linewidths=2.5)
                ax.set_xticks(range(len(treat_list)))
                ax.set_xticklabels([short_treat.get(t, t[:4]) for t in treat_list], fontsize=8)
                ax.set_title(feat.replace("-SUMMARYCustomAlgos.tsv", "").replace(".txt", "")[:26], fontsize=8)
                ax.tick_params(labelsize=7)
            origin_txt = f"blue band = {origin} origin;  " if origin else ""
            title = (f"{strain}: control \u2192 treatments (points = images, dash = median).  "
                     f"{origin_txt}gold band = {dest} destination.")
            suffix = f"band_{strain}"

        elif args.mode == "predclass":
            pred_classes = sorted(sm["pred"].unique())
            rng = np.random.default_rng(0)
            for k, feat in enumerate(top):
                ax = axes[k]
                for j, t in enumerate(treat_list):
                    ids = sm[sm["treat"] == t].index
                    for iid in ids:
                        c = meta.loc[iid, "pred"]
                        jitter = rng.uniform(-0.28, 0.28)
                        ax.scatter([j + jitter], [rf.loc[iid, feat]],
                                   color=class_colour.get(c, "gray"),
                                   marker=class_marker.get(c, "o"),
                                   s=22, alpha=0.7, edgecolors="none")
                ax.set_xticks(range(len(treat_list)))
                ax.set_xticklabels([short_treat.get(t, t[:4]) for t in treat_list], fontsize=8)
                ax.set_title(feat.replace("-SUMMARYCustomAlgos.tsv", "").replace(".txt", "")[:26], fontsize=8)
                ax.tick_params(labelsize=7)
            leg = [Line2D([0], [0], marker=class_marker[c], color="w",
                          markerfacecolor=class_colour[c], markeredgecolor="none",
                          label=c, markersize=9) for c in pred_classes]
            fig.legend(handles=leg, loc="upper right", fontsize=9, title="predicted class")
            title = f"{strain}: each treated image by PREDICTED class (colour + symbol, jittered; top {args.n} features)"
            suffix = f"predclass_{strain}"

        else:  # perdest -- each treatment's band = training range of ITS majority predicted class
            # majority predicted class per treatment (within this strain)
            maj_by_treat = {}
            for t in treat_list:
                g = sm[sm["treat"] == t]
                if len(g):
                    maj_by_treat[t] = g["pred"].value_counts().index[0]
            # origin = this strain's own true class (its --st-- value)
            origin = strain if strain in set(train[args.label_col].unique()) else None
            origin_vals = train[train[args.label_col] == origin] if origin else None
            # distinct colour per destination class that appears
            dest_classes = sorted(set(maj_by_treat.values()))
            dcol = {c: class_colour.get(c, "gold") for c in dest_classes}

            palette = ["#333333", "#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#8c564b"]
            tcol = {t: palette[j % len(palette)] for j, t in enumerate(treat_list)}

            half = 0.4  # half-width of each per-treatment band around its x position
            for k, feat in enumerate(top):
                ax = axes[k]
                # origin reference band spanning the whole panel (steel blue)
                if origin_vals is not None:
                    o = origin_vals[feat].astype(float)
                    ax.axhspan(o.quantile(0.25), o.quantile(0.75), color="steelblue", alpha=0.12)
                    ax.axhline(o.median(), color="steelblue", lw=1, ls=":")
                # per-treatment destination band drawn locally around each x
                for j, t in enumerate(treat_list):
                    dclass = maj_by_treat.get(t)
                    if dclass is not None:
                        dv = train[train[args.label_col] == dclass][feat].astype(float)
                        lo, hi = dv.quantile(0.25), dv.quantile(0.75)
                        ax.fill_between([j - half, j + half], [lo, lo], [hi, hi],
                                        color=dcol[dclass], alpha=0.18, linewidth=0)
                        ax.hlines(dv.median(), j - half, j + half,
                                  color=dcol[dclass], lw=1.2, linestyles="--")
                    ids = sm[sm["treat"] == t].index
                    v = rf.loc[ids, feat].astype(float)
                    ax.scatter([j] * len(v), v, alpha=0.4, s=12, color=tcol.get(t, "gray"))
                    if len(v):
                        ax.scatter([j], [v.median()], marker="_", s=400,
                                   color=tcol.get(t, "gray"), linewidths=2.5)
                ax.set_xticks(range(len(treat_list)))
                ax.set_xticklabels([short_treat.get(t, t[:4]) for t in treat_list], fontsize=8)
                ax.set_title(feat.replace("-SUMMARYCustomAlgos.tsv", "").replace(".txt", "")[:26], fontsize=8)
                ax.tick_params(labelsize=7)
            # legend: which colour = which destination class + the origin band
            from matplotlib.patches import Patch
            leg = [Patch(facecolor=dcol[c], alpha=0.4, label=f"dest: {c}") for c in dest_classes]
            if origin is not None:
                leg.append(Patch(facecolor="steelblue", alpha=0.3, label=f"origin: {origin}"))
            fig.legend(handles=leg, loc="upper right", fontsize=9, title="reference bands")
            otxt = f"origin (blue) = {origin};  " if origin else ""
            title = (f"{strain}: control \u2192 treatments.  {otxt}"
                     f"each treatment's band = training range of ITS majority predicted class.")
            suffix = f"perdest_{strain}"

        for k in range(len(top), len(axes)):
            axes[k].axis("off")

        fig.suptitle(title, fontsize=13, y=1.0)
        plt.tight_layout()
        out = os.path.join(args.out_dir, f"treatment_{suffix}")
        for ext in ("png", "svg"):
            plt.savefig(f"{out}.{ext}", dpi=100, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out + ".png/.svg")

    # ---- majority-class summary: per strain x treatment, vs that strain's true class ----
    print("\n=== majority predicted class per strain x treatment ===")
    print("(true class per strain shown in brackets; \u2260 flags a shift away from it)")
    for strain in sorted(meta["strain"].unique()):
        sm = meta[meta["strain"] == strain]
        true_cls = sm["true"].iloc[0]
        for t in order_treatments(sorted(sm["treat"].unique()), pref):
            g = sm[sm["treat"] == t]
            vc = g["pred"].value_counts()
            m, frac = vc.index[0], vc.iloc[0] / vc.sum()
            label = "CTRL" if (pref and t == pref[0]) else t
            flag = "  (= true)" if m == true_cls else f"  (\u2260 true)"
            print(f"  {strain:6} [{true_cls:>8}]  {label:16} -> majority {m} ({frac*100:.0f}%){flag}")


if __name__ == "__main__":
    main()