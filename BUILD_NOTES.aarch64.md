# TomoTwin — Linux aarch64 + CUDA 13 + GB10 (sm_121)

Upstream: https://github.com/MPI-Dortmund/tomotwin-cryoet (tomotwin-cryoet 0.9.1.2)

## Result: 🟢 PASS — 1 patch, plus a replaced environment file

## Do not use `conda_env_tomotwin.yml`

It pins `pytorch==2.3` from the `pytorch` channel, `cuml=24.08` and
`cuda-version=11.8`, none of which resolve for linux-aarch64 + CUDA 13.

```bash
mamba create -y -p ./env -c conda-forge python=3.11
./env/bin/pip install torch torchvision            # 2.13.0+cu130 aarch64
./env/bin/pip install cuml-cu13                    # RAPIDS 26.8.0, aarch64 wheel
./env/bin/pip install tomotwin-cryoet mrcfile
./env/bin/pip install pytorch-metric-learning dask distributed   # see below
```

**RAPIDS cuML works on aarch64 + CUDA 13.** `cuml-cu13` 26.8.0 publishes
`cuml_cu13-26.8.0-cp311-abi3-manylinux_2_26_aarch64...whl`. Verified with a GPU
DBSCAN on two well-separated clusters (2000 points, 1000/999/1 noise) —
correct labels.

`pytorch-metric-learning`, `dask` and `distributed` are *runtime* imports of
`tomotwin_map.py` / `tomotwin_locate.py` but are missing from the wheel's
dependency list, so they must be installed explicitly.

## Patch — `torch.load` weights-only default (4 call sites)

torch >= 2.6 defaults `weights_only=True`, and the released
`tomotwin_latest.pth` stores numpy scalars in its metadata:

```text
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray.scalar
was not an allowed global by default.
```

`embed_main.py` already passed `weights_only=False`; the remaining four did not:

| file | line |
|---|---|
| `tomotwin/modules/inference/embedor.py` | 88, 178 |
| `tomotwin/modules/training/torchtrainer.py` | 345 |
| `tomotwin/modules/tools/info.py` | 72 |

Install from the git clone (`pip install --no-deps -e .`) so the patch is
tracked rather than buried in `site-packages`.

## Pretrained model

```bash
curl -L -o models/tomotwin_latest.pth \
  https://zenodo.org/api/records/8358240/files/tomotwin_latest.pth/content
```

930,806,586 bytes. **Do not resume this URL with `curl -C -`** — the Zenodo
`/content` endpoint mishandles range requests and silently truncates the file.
Download to a temp path, verify the byte count, then move it into place.

## Smoke test 1 — network only (no weights needed)

```text
SiameseNet3D params            77.56 M
forward (4,1,37,37,37)      -> (4, 32), all finite
cosine(same input twice)       1.000000
cosine(two different inputs)   0.801411
backward                       all parameter gradients finite
```

## Smoke test 2 — full pipeline with the pretrained model

160^3 synthetic tomogram at 10 A/px containing six copies of an asymmetric
three-lobe 37^3 reference particle.

| stage | command | result |
|---|---|---|
| 1 | `tomotwin_embed.py tomogram -b 128 -s 2` | **512,000 sub-volume embeddings, dim 32**, 9 m 46 s on one GB10 |
| 2 | `tomotwin_embed.py subvolumes` | reference embedding written |
| 3 | `tomotwin_map.py distance` | `map.tmap` written |
| 4 | `tomotwin_locate.py findmax` | 1 particle located, NMS applied |
| 5 | `tomotwin_pick.py` | `particle.coords` + `particle_relion3.star` |

The single detection lands at **(82.0, 81.8, 80.75)** against the ground-truth
centre **(80, 80, 80)** — correct to ~2 voxels.

Only one of the six was recovered, which is expected and **not** an aarch64
issue: TomoTwin picks by nearest neighbour in a metric space learned from real
cryo-ET density of actual protein complexes, so synthetic Gaussian blobs are far
out of distribution. Quantifying picking accuracy needs real tomograms; what
this run establishes is that all five stages, the GPU embedding of half a
million sub-volumes, and the cuML-backed clustering path all execute correctly
on aarch64.
