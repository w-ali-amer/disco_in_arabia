/* exp137_enum.c -- doc 137 E2 (amended 3b3aed3): weighted variant of exp132_enum.c for skewed class priors.
   usage: exp137_enum K C M rep_lo rep_hi stride j priorfile
     unreduced only (exp132 mode 1): p0 over all labellings, p0 <= p1 <= ... (message-relabelling symmetry only),
     p0 in rep_lo..rep_hi-1, p1 restricted to p1 % stride == j.
     priorfile: C doubles p(class), whitespace separated.  Tuple weight w(x) = prod_i p(x_i) / Z / K.
   prints: V p0 p1 [p2 p3] count N L   (V = sum_x w(x) max_s matches, %.17g; first argmax in loop order) */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int K, C, M, L, N;
static uint8_t *F; /* L x N matches of labelling p on each tuple */
static double *W;

static int digit(int p, int v) { for (int i = 0; i < v; i++) p /= K; return p % K; }

static inline double wsum(const uint8_t *a, const uint8_t *b) {
    double s = 0.0;
    for (int t = 0; t < N; t++) { uint8_t m = a[t] > b[t] ? a[t] : b[t]; s += W[t] * (double)m; }
    return s;
}

static inline void vmax(uint8_t *o, const uint8_t *a, const uint8_t *b) {
    for (int t = 0; t < N; t++) o[t] = a[t] > b[t] ? a[t] : b[t];
}

int main(int argc, char **argv) {
    if (argc != 9) { fprintf(stderr, "usage\n"); return 2; }
    K = atoi(argv[1]); C = atoi(argv[2]); M = atoi(argv[3]);
    int rep_lo = atoi(argv[4]), rep_hi = atoi(argv[5]), stride = atoi(argv[6]), jj = atoi(argv[7]);
    if (M != 2 && M != 4) { fprintf(stderr, "M must be 2 or 4\n"); return 2; }
    double pr[16]; FILE *fp = fopen(argv[8], "r"); if (!fp) { fprintf(stderr, "prior file\n"); return 2; }
    for (int i = 0; i < C; i++) if (fscanf(fp, "%lf", &pr[i]) != 1) { fprintf(stderr, "prior read\n"); return 2; }
    fclose(fp);
    L = 1; for (int i = 0; i < C; i++) L *= K;
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
    W = malloc(sizeof(double) * N); double Z = 0.0;
    for (int x = 0; x < N; x++) { double w = 1.0; for (int i = 0; i < K; i++) w *= pr[tup[x * K + i]]; W[x] = w; Z += w; }
    for (int x = 0; x < N; x++) W[x] /= Z * K;
    F = calloc((size_t)L * N, 1);
    for (int p = 0; p < L; p++)
        for (int x = 0; x < N; x++) {
            int m = 0;
            for (int i = 0; i < K; i++) if (digit(p, tup[x * K + i]) == i) m++;
            F[(size_t)p * N + x] = (uint8_t)m;
        }
    if (rep_hi > L) rep_hi = L;
    uint8_t *c1 = malloc(N), *c2 = malloc(N);
    double best = -1.0; int arg[4] = {-1, -1, -1, -1}; uint64_t count = 0;
    for (int p0 = rep_lo; p0 < rep_hi; p0++) {
        const uint8_t *f0 = F + (size_t)p0 * N;
        for (int p1 = p0; p1 < L; p1++) {
            if (p1 % stride != jj) continue;
            if (M == 2) {
                double s = wsum(f0, F + (size_t)p1 * N); count++;
                if (s > best) { best = s; arg[0] = p0; arg[1] = p1; }
                continue;
            }
            vmax(c1, f0, F + (size_t)p1 * N);
            for (int p2 = p1; p2 < L; p2++) {
                vmax(c2, c1, F + (size_t)p2 * N);
                for (int p3 = p2; p3 < L; p3++) {
                    double s = wsum(c2, F + (size_t)p3 * N);
                    if (s > best) { best = s; arg[0] = p0; arg[1] = p1; arg[2] = p2; arg[3] = p3; }
                }
                count += (uint64_t)(L - p2);
            }
        }
    }
    if (M == 2) printf("%.17g %d %d %llu %d %d\n", best, arg[0], arg[1], (unsigned long long)count, N, L);
    else printf("%.17g %d %d %d %d %llu %d %d\n", best, arg[0], arg[1], arg[2], arg[3], (unsigned long long)count, N, L);
    return 0;
}
