# AdaptiveCpp samples

This directory contains larger, self-contained sample applications that
demonstrate how SYCL can be used together with other APIs. The samples are not
built as part of the AdaptiveCpp build; they are standalone CMake projects that
require an AdaptiveCpp installation:

```
cd samples
cmake -B build -DCMAKE_PREFIX_PATH=/path/to/adaptivecpp/install
cmake --build build
```

Samples whose dependencies (see the individual READMEs) cannot be found are
skipped automatically during configuration.

| Sample | Description |
| ------ | ----------- |
| [`ocean`](ocean/README.md) | FFT based ocean surface simulation in SYCL, rendered with Vulkan |
