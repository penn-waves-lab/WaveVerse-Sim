extern "C" __global__
void pano_corr(const float2* x, const float2* w, float2* y,
               int K, int N, int H) {
    int q = blockDim.x * blockIdx.x + threadIdx.x;
    if (q >= K * N) return;
    int k = q / N, r = q - k * N;
    double re = 0.0, im = 0.0;
    for (int j = 0; j < H; ++j) {
        float2 a = x[(k + j) * N + r];
        float2 b = w[j * N + r];
        re += (double)a.x * b.x - (double)a.y * b.y;
        im += (double)a.x * b.y + (double)a.y * b.x;
    }
    y[q] = make_float2((float)re, (float)im);
}
