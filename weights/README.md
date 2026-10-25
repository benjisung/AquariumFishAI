# Model checkpoints

| File | Role | SHA-256 |
|---|---|---|
| `reef_ai_yolov12n_v1_controlled.pt` | Initial controlled-tank model; included for comparison | `a5e6c4d78d830aa72c368734b747f09afd1e073b8c56fa6368b0d4e0a8ddf3a6` |
| `reef_ai_yolov12n_v2_full_tank.pt` | Full-tank adaptation model; recommended demo checkpoint | `bfe58b9d8e8b1c1091ed148a2f80b384111f823d76629bb3fce85dea2addeb57` |

Both public checkpoint copies have normalized path metadata: their model and EMA tensor-state digests were verified unchanged against the source checkpoints after metadata cleanup. This removes local training paths while preserving all model parameters. The v2 checkpoint is the primary model for demonstrations in this repository. The v1 checkpoint is retained to make the published comparison concrete.

The numeric class 7 remains compatible with historical internal metadata. For public display and manuscript reporting this project identifies it as *Monodactylus argenteus* (Silver moony); see `../docs/SPECIES_MAPPING.md`.

No license for checkpoint redistribution is declared in this repository. Confirm applicable YOLOv12/Ultralytics terms before redistribution or commercial use.
