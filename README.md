# Generating Ancient Script Symbols with Diffusion Models

Unconditional diffusion models (DDPM) trained to generate new symbols in two ancient writing systems, as a research assignment for a Generative AI Fundamentals course. The project compares three cases under identical settings:

| Case | Source | Signs |
|---|---|---|
| Egyptian hieroglyphs (normal) | Unicode block *Egyptian Hieroglyphs Extended-A* (U+13460–143FA) | 3,995 |
| Egyptian hieroglyphs (bold) | The same signs with thickened strokes | 3,995 |
| Cuneiform | Unicode blocks *Cuneiform*, *Cuneiform Numbers and Punctuation*, *Early Dynastic Cuneiform* (U+12000–12543) | about 1,230 |

The normal and bold hieroglyph cases use the same signs and the same train/test split, so stroke weight is the only difference between them.

## Research questions

1. **Dataset size:** how many symbols are needed to generate high-quality output?
2. **Quality:** does the model work on these scripts, and are the generated symbols visually meaningful?
3. **Modifications:** do changes to the architecture or noise schedule improve the results?

## Repository contents

| File | Purpose |
|---|---|
| `Diffusion_Egyptian_Hieroglyphs_normal_bold.ipynb` | Full pipeline for the two hieroglyph cases |
| `Diffusion_Cuneiform.ipynb` | Full pipeline for the Cuneiform case |
| `README.md` | This file |

<!-- Rename the entries above if your notebook file names differ. Add the report (PDF) here once it is written. -->

Each notebook runs end to end: dataset extraction, train/test split, training, evaluation, and report figures. The two notebooks write to the same project folder, so the final tables and figures cover all three cases.

## Method

**Data.** Each sign is cropped from the official Unicode code-chart PDF, centred on a white 128×128 canvas and saved as a grayscale PNG. Reserved (unassigned) chart cells are skipped. The bold case applies a 3-pixel min-filter followed by a light blur. Ten percent of the signs in each script are held out as a test set and never used for training.

**Model.** A `UNet2DModel` from Hugging Face `diffusers` (about 77M parameters, six resolution levels, self-attention at the 8×8 level), trained to predict noise with a 1,000-step linear DDPM schedule. Sampling uses DPM-Solver++ with 50 steps.

**Training.** Every run uses the same budget: 10,000 steps, batch size 32, AdamW at 1e-4 with warm-up and cosine decay, mixed precision. Fixing the number of steps, not epochs, keeps runs on small datasets from being under-trained.

**Experiments.**

| Question | Runs |
|---|---|
| Q1 | Baseline model on nested subsets: 250, 500, 1,000, 2,000 and all signs (hieroglyphs); 125, 250, 500 and all signs (Cuneiform) |
| Q2 | Detailed evaluation of the best run per case |
| Q3 | Changes to the full-data baseline: cosine noise schedule, EMA weights, and (hieroglyphs) further variants defined in `Q3_VARIANTS`; plus quality against the number of sampling steps |

**Metrics.** Each run generates 500 images from a fixed seed and compares them with the held-out signs.

| Metric | Meaning |
|---|---|
| KID, FID | Distance between generated and real held-out images (lower is better). A real-vs-real score is reported as the floor. |
| Copy rate | Share of generated images that sit closer to a training sign than 95% of unseen real signs do, meaning near-copies |
| NN ratio | Median distance to the nearest training sign, generated vs. real unseen signs (about 1 means as novel as real signs) |
| Shape statistics | Ink ratio and connected-component count, generated vs. real |
| Failure rate | Blank or pure-noise outputs |

A blind human-evaluation pack (mixed real and generated signs with a hidden answer key) is also produced.

## How to run

1. Open a notebook in Google Colab and select a GPU runtime.
2. Download the code-chart PDFs from [unicode.org/charts](https://www.unicode.org/charts/) and place them in `My Drive/glyph_diffusion/` (or the Colab working folder), using the file names in each notebook's configuration cell:
   - `Egyptian Heiroglyphs.pdf` (Egyptian Hieroglyphs Extended-A)
   - `cuneiform-unicode.pdf`, `cuneiform-numbers-unicode.pdf`, `cuneiform-early-dynastic.pdf`
3. Run all cells. Results are saved to Google Drive. Finished runs are skipped, and interrupted runs resume from the last checkpoint, so the notebook can be re-run across sessions.
4. Run the other notebook the same way. Whichever notebook runs last produces the combined three-case figures and tables.

**Compute.** A 10,000-step run takes about 26 minutes on an A100 and about 108 minutes on an L4. The run plan can be shortened in the "Experiment plan" section of each notebook. Keep `TRAIN_STEPS` the same for every run so results stay comparable.

## Outputs

```
glyph_diffusion/
  data/      extracted PNGs, metadata CSVs, train/test splits
  runs/      one folder per run: config, loss curve, progress snapshots, samples, metrics
  figures/   report figures (dataset, Q1, Q2, Q3)
  report/    results.csv, results tables, key numbers, human-evaluation pack
```

The chart PDFs, extracted images and trained models are not included in this repository. The notebooks regenerate them.

## Results

<!-- Add a link to the report and two or three headline findings here. -->

See the project report: *link to be added*.

## Acknowledgements

- The training code builds on the Chinese-character diffusion notebook provided in the course.
- Glyph images come from the Unicode Consortium's code charts. The charts and their fonts are subject to the [Unicode Terms of Use](https://www.unicode.org/copyright.html).
- Built with [Hugging Face Diffusers](https://github.com/huggingface/diffusers), PyTorch, TorchMetrics and PyMuPDF.
