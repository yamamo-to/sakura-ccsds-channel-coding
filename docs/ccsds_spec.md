# CCSDS 131.0-B Channel Coding Specifications

## 1. Reed-Solomon Code (RS)
- **Standard:** CCSDS (255, 223) Reed-Solomon code over GF(2^8)
- **Field Generator Polynomial:** $p(x) = x^8 + x^7 + x^2 + x + 1$
- **Code Generator Polynomial:** $g(x) = \prod_{j=112}^{143} (x - \alpha^j)$
- **Interleaving:** Depth $I = 1$ to $5$ configurable.
- **Dual Basis / Conventional Basis:** Default to Conventional Representation.

## 2. Convolutional Code (CONV)
- **Constraint Length ($K$):** 7
- **Code Rate:** $R = 1/2$
- **Generator Polynomials:**
  - $G_1 = 121_8$ (1010001_2) (Inverting)
  - $G_2 = 133_8$ (1011011_2) (Non-inverting)
- **Puncturing Schemes:** Support $2/3, 3/4, 5/6, 7/8$ if enabled.

## 3. Turbo Code
- **Code Rates:** $1/2, 1/3, 1/4, 1/6$
- **Block Lengths ($K$):** 1784, 3568, 7136, 8920, 16384 bits
- **Constituent Code:** Recursive Systematic Convolutional (RSC) with $K=5$
  - Feedback polynomial: $g_0 = 10011_2$ ($23_8$)
  - Forward polynomial: $g_1 = 11011_2$ ($33_8$)
- **Interleaver:** CCSDS Permutation Algorithm (using $k_1, k_2$ table constants).
- **Decoder:** Log-MAP or Max-Log-MAP algorithm (Max 10 iterations).
