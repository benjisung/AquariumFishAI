# Data and model provenance

## Controlled development split

The v1 dataset has 349 training and 70 validation images (419 total). The split script applied a seed-42 random image-level split to the first 346 reviewed frames (276/70), then placed all 73 later auto-labeled frames in training. These frames came from the same controlled-box recording extracted every two seconds. Thus neighboring times and recurring fish/background can appear in both sets. Call it same-recording development validation.

## Full-tank adaptation

The v2 adaptation images came from a 60-second full-tank session recorded on 2026-08-02, based on the video overlay. Forty-eight images were used for training and twelve for development validation. The model was initialized from v1 and fine-tuned for 60 epochs. The old controlled v1 images, including their old validation subset, were also moved into v2 training.

## Final evaluation source

The independent test came from a distinct 6,033.4-second video recorded on 2026-08-22. Its frames were sampled across the recording at roughly 102.09-second intervals. The setup is the same aquarium deployment with a materially similar frontal view; isolation boxes and some top equipment differ. Both videos display the logical camera label `Camera2`. The physical camera device identity is not established.

The class frequencies observed in adaptation annotations and final-test annotations differ. The annotations do not identify individual fish, so they cannot establish which individuals remained, entered, or left the aquarium.

## Contamination checks

- The 60 final-test images had no SHA-256 matches among the 419 v1 images or the 479 v2 dataset images.
- The v1 and v2 checkpoint timestamps predate the final recording.
- Existing dataset preparation scripts do not reference the final test source.
- The frozen-test protocol excludes the final set from training, model selection, and threshold selection.

These checks found no contamination in the files and records audited. They are evidence-based checks, not proof of undocumented actions outside the available records.

## Source paths withheld

Local drive paths and user account names are omitted. Source records, hashes for private images, timestamp-by-timestamp manifests, and raw materials are not published in this repository.
