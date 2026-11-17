# Mixed Precision

## How a number is stored

A float is stored as: **sign, exponent, and mantissa**.

    value = ± 1.XXXXXX × 2^E

- **Mantissa (the X's):** the number of significant digits after the decimal point.
  It sets the **precision**. More mantissa bits = more digits, less rounding error.
- **Exponent (E):** shifts the decimal point left or right.
  It sets the **dynamic range**. More exponent bits = larger and smaller representable numbers.
- **Sign:** 1 bit (positive or negative).

## Formats

| Format | Sign | Exponent | Mantissa | Max value | Smallest normal | Relative precision |
|--------|------|----------|----------|-----------|-----------------|--------------------|
| fp32   | 1    | 8        | 23       | ~3×10³⁸   | ~10⁻³⁸          | ~10⁻⁷              |
| fp16   | 1    | 5        | 10       | 65,504    | ~6×10⁻⁵         | ~10⁻³              |
| bf16   | 1    | 8        | 7        | ~3×10³⁸   | ~10⁻³⁸          | ~10⁻²              |

- **bf16** = fp32 with the mantissa truncated: same range, less precision.
- **fp16** = more precision than bf16, but a small range.

## Why use 16 bits

- **Memory:** half the bytes for activations, so larger batches fit.
- **Speed:** tensor cores run 16-bit matmuls much faster than 32-bit ones.

## Why it works

1. The minibatch gradient is already noisy; rounding error is small compared to that noise.
2. Rounding errors are roughly random and cancel out in sums.
3. The tensor core multiplies in 16 bits but **accumulates in fp32**.

## Where 16 bits breaks, and the fix

| Problem | Cause | Fix |
|---------|-------|-----|
| Gradient becomes 0 (underflow) | fp16 floor ~6×10⁻⁸ | loss scaling (or use bf16) |
| Weight update vanishes | short mantissa: 0.02 + 6e-9 = 0.02 | fp32 master weights |
| Softmax, layernorm, loss inaccurate | sensitive sums and exp | autocast keeps them in fp32 |

## Loss scaling (fp16 only)

1. Multiply the loss by S (e.g., 2¹⁵). By the chain rule, all gradients get multiplied by S.
2. Small gradients now fit in fp16's range.
3. Before the update: `unscale_` divides by S in fp32.
4. If `inf`/`nan` appears: skip the step and reduce S (`GradScaler` does this).

## bf16 vs fp16

- **bf16:** same exponent as fp32, no underflow, **no scaler needed**. Standard on Ampere+ GPUs (A100, L4, H100).
- **fp16:** needs `GradScaler`. Used on T4/V100.

## Code (bf16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    (loss / GRAD_ACCUM).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    opt.step(); opt.zero_grad(set_to_none=True)

## Summary

Spend precision only where it matters (accumulating sums, updating weights) and save it where noise is already tolerated (the products inside matmuls).