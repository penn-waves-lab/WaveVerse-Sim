extern "C" __global__
void mix_kernel(const float* __restrict__ x,        // [K, P]      beat frequency, cycles per sample
                const float2* __restrict__ W,       // [K, C, P]   per-channel path weight (sample-independent phase folded in)
                float2* __restrict__ S,             // [K, SPLIT, C, N] partial sums
                int P, int C, int N, int SPLIT)
{
    const int k = blockIdx.x, s = blockIdx.y, n = threadIdx.x;      // position, path split, sample
    const int p0 = (int)(((long long)P * s) / SPLIT), p1 = (int)(((long long)P * (s + 1)) / SPLIT);
    extern __shared__ float sh[];
    float*  shx = sh;                                              // [TILE]
    float2* shW = (float2*)(sh + blockDim.x);                      // [C][TILE]
    float2 acc[8];
    for (int c = 0; c < 8; ++c) acc[c] = make_float2(0.f, 0.f);
    const float* xk = x + (long long)k * P;
    const float2* Wk = W + (long long)k * C * P;
    for (int base = p0; base < p1; base += blockDim.x) {
        const int tile = min((int)blockDim.x, p1 - base);
        __syncthreads();
        if (n < tile) {
            shx[n] = xk[base + n];
            for (int c = 0; c < C; ++c) shW[c * blockDim.x + n] = Wk[(long long)c * P + base + n];
        }
        __syncthreads();
        if (n < N) {
            for (int i = 0; i < tile; ++i) {
                float u = shx[i] * (float)n; u -= floorf(u);          // phase in cycles, mod 1
                float sn, cs; sincospif(2.0f * u, &sn, &cs);
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    if (c < C) {
                        float2 w = shW[c * blockDim.x + i];
                        acc[c].x += w.x * cs - w.y * sn;
                        acc[c].y += w.x * sn + w.y * cs;
                    }
                }
            }
        }
    }
    if (n < N) for (int c = 0; c < C; ++c)
        S[(((long long)k * SPLIT + s) * C + c) * N + n] = acc[c];
}

extern "C" __global__
void prepare_weights(const float2* alpha, const float* tau,
                     float2* W, float* x, int P, int C, long long count,
                     double slope, double t0, double dt) {
    long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count) return;
    long long k = q / P;
    int p = q - k * P;
    double t = (double)tau[q];
    const double two_pi = 6.283185307179586476925286766559;
    double phase = two_pi * (-0.5*slope*(t*t) + slope*t*t0);
    float ph = (float)(phase - floor(phase / two_pi) * two_pi);
    float sn, cs;
    sincosf(ph, &sn, &cs);
    x[q] = (float)(slope*t*dt);
    for (int c = 0; c < C; ++c) {
        long long i = (k*C + c)*P + p;
        float2 a = alpha[i];
        a.y = -a.y;
        W[i] = make_float2(a.x*cs - a.y*sn, a.x*sn + a.y*cs);
    }
}
