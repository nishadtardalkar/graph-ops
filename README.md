# graph-ops
 
 
# nn-ops
Ops to be compiled using Clang-AVX2

Every Op uses:
<ul>
<li>Multithreading (Some ops have #define THREADS to limit number of threads in some cases, change that to your needs.)</li>
<li>SIMD AVX2 (Do rewrite for AVX512 if you are lucky enough to experience 512bit vectors)</li>
<li>FMA instructions (So dont forget -fma while compiling)</li>
</ul><br>
&nbsp&nbsp&nbsp to make things as fast as possible.
<br><br>
<b>I'm compiling from VS2019 with this .bat file: (%1 is the file name to be compiled)</b><br>

```ruby
@echo Building %1
clang.exe -c %1.cpp -fopenmp -msse -msse2 -mavx -mavx2 -mfma -O3 -o %1.o
clang.exe -shared -v -fopenmp -msse -msse2 -mavx -mavx2 -mfma -O3 -o %1.dll %1.o
@echo DONE
```


# gpu-nn-ops
 Highly optimized GPU ops to be compiled with NVCC on your machine.<br>
 
<b>Ops try to use:</b>
<ul>
 <li>Shared memory</li>
 <li>Registers</li>
 <li>float4 loads and stores</li>
 <li>warp and half warp optimized loads</li>
 <li>256 max threads per block to support GPUs other than Nvidia</li>
</ul>
 
<b>Current state of the repo :</b><br>
<ul>
 <li>sgemm</li>
 <li>sgemm strided</li>
 <li>sgemm strided k-batched</li>
 <li>conv2d winograd4x4 forward and backward transforms</li>
 <li>relu activation</li>
 <li>maxpooling2d</li>
 <li>moment and rms gradient update</li>
 <li>mallocs and memsets</li>
 <li>squared difference</li>
 <li>mean</li>
 <li>subtract multiply</li>
 <li>broadcast multiply</li>
 <li>total sum</li>
 <li>strided sum</li>
 <li>add</li>
 <li>tiled add</li>
</ul>
