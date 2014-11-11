__kernel void convolution(__global float * first, __global float * second,
                          __global float * out, int sizeFirst, int sizeMask)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= sizeFirst || j >= sizeFirst)
        return;

    float ans = 0.0;
    int HM = (sizeMask - 1) / 2;

    for (int k = -HM; k <= HM; ++k) {
        for (int l = -HM; l <= HM; ++l) {
            if (i + k >= 0 && j + l >= 0 && i + k < sizeFirst && j + l < sizeFirst) {
                ans += first[(i + k) * sizeFirst + j + l] * second[(k + HM) * sizeMask  + l + HM ];
            }
        }
    }
    out[i * sizeFirst + j] = ans;
}