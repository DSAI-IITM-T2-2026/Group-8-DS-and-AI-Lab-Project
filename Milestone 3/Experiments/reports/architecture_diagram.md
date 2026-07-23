# Milestone 3 — Architecture Diagram

## End-to-end pipeline

```mermaid
flowchart TB
  subgraph sources [GCS anonymous]
    FIRMS[FIRMS daily GeoTIFF ~1km]
    S2[S2 CSV ~5-day windows]
    S5P[S5P CSV daily]
    ERA5[ERA5 monthly ZIP-NetCDF]
    DEM[DEM static terrain]
  end

  subgraph prep [Local preprocessing]
    AOI[AOI lock ERA5-bounded]
    Ref[FIRMS reference grid]
    Align[S2 forward-fill windows to days]
    Fuse[Regrid or join onto FIRMS grid]
    Sample[Fire-centered 64x64 patches 2025]
    Split[Temporal 70/15/15]
    Norm[Z-score from train only]
  end

  subgraph models [Train from scratch — no pretrained wildfire weights]
    Floor[Always no-fire floor]
    XGB[HistGradientBoosting sklearn]
    DL[ConvLSTM + U-Net via PyTorch]
  end

  subgraph eval [Evaluation]
    Metrics[Precision Recall F1 Dice AUC-PR]
    Tune[Random search 8 trials]
    Plots[PR curves]
  end

  FIRMS --> AOI
  S2 --> AOI
  S5P --> AOI
  ERA5 --> AOI
  DEM --> AOI
  AOI --> Ref
  Ref --> Fuse
  FIRMS --> Fuse
  S2 --> Align --> Fuse
  S5P --> Fuse
  ERA5 --> Fuse
  DEM --> Fuse
  Fuse --> Sample
  Sample --> Split
  Split --> Norm
  Norm --> Floor
  Norm --> XGB
  Norm --> DL
  DL --> Tune
  Floor --> Metrics
  XGB --> Metrics
  Tune --> Metrics
  Metrics --> Plots
```

## Primary model (defined in code, trained locally)

```mermaid
flowchart LR
  In["Input B x 7 x 30 x 64 x 64"] --> ConvLSTM["ConvLSTM x2 hidden=32"]
  ConvLSTM --> Last["Last hidden B x 32 x 64 x 64"]
  Last --> Enc["U-Net encoder + skips"]
  Enc --> Bot["Bottleneck"]
  Bot --> Dec["U-Net decoder + skips"]
  Dec --> Out["Logits B x 1 x 64 x 64"]
  Out --> Sig["Sigmoid risk map"]
```

## Libraries vs downloaded models

| Piece | What you install | Pretrained wildfire weights? |
|---|---|---|
| Tree baseline | `sklearn` HistGradientBoosting | No — trains on our patches |
| ConvLSTM+U-Net | `pip install torch` + our `src/models/convlstm_unet.py` | No — architecture in repo, train from scratch |
| Checkpoints after train | Saved under `data/processed/checkpoints/` | Created by us, not downloaded |

## Loss / metrics

- Losses: **BCE + Dice** vs **Focal**
- Headline: precision, recall, F1, Dice, AUC-PR (fire class)
- Accuracy only as a caveated secondary number
