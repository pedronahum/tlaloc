# pyptx-generated PTX fixtures (§0.4.357)

Generated from [pyptx](https://github.com/patrick-toulme/pyptx) 0.1.1
(**Apache-2.0**) example kernel builders, frozen here as the foreign-PTX
ingestion corpus for the KPTX bootstrap pipeline
(`normalizePtx → transpile`):

| file | source | shapes |
|------|--------|--------|
| `pyptx_rms_norm.ptx` | `pyptx.examples.ampere.rms_norm.build_rms_norm(256, 512)` | 256×512 f32, v4 loads, shfl.bfly reduction |
| `pyptx_softmax.ptx`  | `pyptx.examples.hopper.softmax.build_softmax(256, 512, arch="sm_80")` | two-pass max+sum, ex2 fold |
| `pyptx_gemm.ptx`     | `pyptx.examples.ampere.gemm.build_gemm(256, 256, 256)` | mma.sync m16n8k16 bf16 tile |

Regenerate with the venv's pyptx if the pin ever moves; the ingestion
test (`PyptxIngestionTest`) treats these bytes as the contract.
