/* exp132_enum.c -- doc 132: exhaustive two-bit (M=4) / one-bit (M=2) search for the which-referent game.
   usage: exp132_enum K C M mode rep_lo rep_hi stride j
     mode 0 = reduced (p0 over class-canonical reps, p1..p_{M-1} sorted over all labellings)
     mode 1 = unreduced (p0 over all labellings, p0 <= p1 <= ... )
     reps rep_lo..rep_hi-1 (indices into the rep list), p1 restricted to p1 % stride == j.
   prints: T p0 p1 [p2 p3] count   (T = best sum over tuples of max_s matches; first argmax in loop order) */
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int K, C, M, L, N, NP;
static uint8_t *F; /* L x NP, row p = matches of labelling p on each tuple, zero padded */

static int digit(int p, int v) { for (int i = 0; i < v; i++) p /= K; return p % K; }

static inline uint32_t summax(const uint8_t *a, const uint8_t *b) {
    uint16x8_t acc = vdupq_n_u16(0);
    for (int t = 0; t < NP; t += 16) acc = vpadalq_u8(acc, vmaxq_u8(vld1q_u8(a + t), vld1q_u8(b + t)));
    return vaddvq_u16(acc);
}

static inline void vmax(uint8_t *o, const uint8_t *a, const uint8_t *b) {
    for (int t = 0; t < NP; t += 16) vst1q_u8(o + t, vmaxq_u8(vld1q_u8(a + t), vld1q_u8(b + t)));
}

int main(int argc, char **argv) {
    if (argc != 9) { fprintf(stderr, "usage\n"); return 2; }
    K = atoi(argv[1]); C = atoi(argv[2]); M = atoi(argv[3]); int mode = atoi(argv[4]);
    int rep_lo = atoi(argv[5]), rep_hi = atoi(argv[6]), stride = atoi(argv[7]), jj = atoi(argv[8]);
    if (M != 2 && M != 4) { fprintf(stderr, "M must be 2 or 4\n"); return 2; }
    L = 1; for (int i = 0; i < C; i++) L *= K;
    /* ordered K-tuples of distinct classes, lexicographic (same order as itertools.permutations) */
    int cap = 1; for (int i = 0; i < K; i++) cap *= (C - i);
    int *tup = malloc(sizeof(int) * cap * K); N = 0;
    int idx[8] = {0};
    for (;;) {
        int ok = 1;
        for (int a = 0; a < K && ok; a++) for (int b = 0; b < a; b++) if (idx[a] == idx[b]) { ok = 0; break; }
        if (ok) { memcpy(tup + N * K, idx, sizeof(int) * K); N++; }
        int pos = K - 1; while (pos >= 0 && ++idx[pos] == C) { idx[pos] = 0; pos--; }
        if (pos < 0) break;
    }
    if (N != cap) { fprintf(stderr, "tuple count\n"); return 3; }
    NP = (N + 15) / 16 * 16;
    F = calloc((size_t)L * NP, 1);
    for (int p = 0; p < L; p++)
        for (int x = 0; x < N; x++) {
            int m = 0;
            for (int i = 0; i < K; i++) if (digit(p, tup[x * K + i]) == i) m++;
            F[(size_t)p * NP + x] = (uint8_t)m;
        }
    /* rep list */
    int *reps = malloc(sizeof(int) * L), nrep = 0;
    if (mode == 1) { for (int p = 0; p < L; p++) reps[nrep++] = p; }
    else {
        for (int p = 0; p < L; p++) { /* class-canonical: digits non-decreasing in class index */
            int ok = 1;
            for (int v = 1; v < C; v++) if (digit(p, v) < digit(p, v - 1)) { ok = 0; break; }
            if (ok) reps[nrep++] = p;
        }
    }
    if (rep_hi > nrep) rep_hi = nrep;
    uint8_t *c1 = aligned_alloc(16, NP), *c2 = aligned_alloc(16, NP);
    uint32_t best = 0; int arg[4] = {-1, -1, -1, -1}; uint64_t count = 0;
    for (int r = rep_lo; r < rep_hi; r++) {
        int p0 = reps[r]; const uint8_t *f0 = F + (size_t)p0 * NP;
        int start = (mode == 1) ? p0 : 0;
        for (int p1 = start; p1 < L; p1++) {
            if (p1 % stride != jj) continue;
            if (M == 2) {
                uint32_t s = summax(f0, F + (size_t)p1 * NP); count++;
                if (s > best) { best = s; arg[0] = p0; arg[1] = p1; }
                continue;
            }
            vmax(c1, f0, F + (size_t)p1 * NP);
            for (int p2 = p1; p2 < L; p2++) {
                vmax(c2, c1, F + (size_t)p2 * NP);
                for (int p3 = p2; p3 < L; p3++) {
                    uint32_t s = summax(c2, F + (size_t)p3 * NP);
                    if (s > best) { best = s; arg[0] = p0; arg[1] = p1; arg[2] = p2; arg[3] = p3; }
                }
                count += (uint64_t)(L - p2);
            }
        }
    }
    if (M == 2) printf("%u %d %d %llu %d %d %d\n", best, arg[0], arg[1], (unsigned long long)count, nrep, N, L);
    else printf("%u %d %d %d %d %llu %d %d %d\n", best, arg[0], arg[1], arg[2], arg[3], (unsigned long long)count, nrep, N, L);
    return 0;
}
