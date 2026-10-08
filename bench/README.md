# bench

Metal performance probes for effect-torch. Run with `bun bench/<file>.ts`.
They need the release Metal addon (see the root README).

| File | Measures |
| --- | --- |
| `metal-throughput.ts` | Op support probes, 4096³ matmul TFLOPS, and a 27M GPT training step in f32 and mixed BF16 |
| `matmul-chain.ts` | Matmul TFLOPS per dtype and size, with 8 matmuls chained in one program |
| `matmul-3d.ts` | 3-D × 2-D matmul against the flattened 2-D form, attention score variants, and full vs row-wise reductions |

## Results on M1 Max, 64 GB

- Matmul 4096³: f32 2.16 TFLOPS, bf16 2.18, f16 2.30. Half precision gives no speedup.
- 27M GPT step, batch 16, seq 256: f32 482 ms (8496 tok/s), mixed BF16 522 ms.
- `bf16` `conv2d` fails: `exact pipeline is not warm`.
- `matmul-3d.ts`: 3-D × 2-D matmul 279 ms against 68 ms when reshaped to 2-D. A full `sum` to a scalar takes 21 ms against 0.7 ms for a row-wise sum.
