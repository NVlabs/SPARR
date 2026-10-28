<!--
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the NVIDIA License.
See the LICENSE file in the repository root for the full license text.
-->

# Third-party notices

The root [LICENSE](LICENSE) covers NVIDIA-authored SPARR material. It does not
replace the licenses of the following vendored components.

| Directory | Upstream | License |
| --- | --- | --- |
| `agentlace/` | [youliangtan/agentlace](https://github.com/youliangtan/agentlace/tree/cf2c337c5e3694cdbfc14831b239bd657bc4894d) | [MIT](agentlace/LICENSE), Copyright (c) 2023 You Liang Tan |
| `serl_launcher/` | [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl) / [SERL](https://github.com/rail-berkeley/serl) | [Apache-2.0](serl_launcher/LICENSE); upstream notice names Jianlan Luo, Charles Xu, and Jeffrey Wu (2024) |

Local modifications are described in [Agentlace's patch notes](agentlace/SPARR_PATCHES.md)
and [SERL's patch notes](serl_launcher/SPARR_PATCHES.md). Existing third-party
attributions are retained; NVIDIA license headers are not applied to these
vendored trees.

## SERL attribution

The bundled `serl_launcher/LICENSE` is copied unchanged from
[HIL-SERL commit c32939bccb65f3b8c43a9f9add3d322d4ab0264a](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/LICENSE).
This reference identifies the license source; the original vendoring commit
was not recorded in SPARR.

The upstream [SERL README](serl_launcher/README.md) credits
[jaxrl_m](https://github.com/dibyaghosh/jaxrl_m) and
[jaxrl_minimal](https://github.com/rail-berkeley/jaxrl_minimal).
The public jaxrl_m project's [MIT notice](serl_launcher/LICENSE.jaxrl_m)
is included unchanged for the code derived from it.
`serl_launcher/serl_launcher/vision/film_conditioning_layer.py` is adapted from
[Google Research's robotics_transformer FiLM layer](https://github.com/google-research/robotics_transformer/blob/master/film_efficientnet/film_conditioning_layer.py),
Copyright 2022 Google LLC, under Apache-2.0. Its original license header is
retained alongside the adaptation attribution.

## Runtime dependencies

Libraries downloaded or installed by the Docker build retain their respective
licenses. The dependency versions and source revisions are recorded in
`docker/phase1/source-lock-phase1.env` and
`docker/phase2/source-lock-phase2.env`.
