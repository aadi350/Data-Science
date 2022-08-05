A thread is a single "process"
Blocks of threads
Warp is a collection of blocks
# What is a page table
Unified memory access streamlines operations on device and on-host, maximize data access speed by transparent data migrating  
UM eliminates need for explicit data movement via `cudaMemcp*()`

`cudaMemPrefetchhAsync` prefetches memory to the specified destination device

“Page Fault” (fault)
The addressed page is not present in memory, the corresponding Page Table
entry is null, or a violation of the paging protection mechanism has occurred.

UM where all processos see a coherent memoy image with common address space, no need fo explicit memory copy calls

UM is for writing simple code, doesn't necessarily result in a speed increase, no need for explicit memory transfers between host and device, automatic migration

`void* malloc (size_t size);` allocates a block of `size` memory, returning a pointer to the beginning of this block. `cudaMalloc()` does the same for linear memory, typically copy from host to device using `cudaMemcpy()`.

UM decouples memory and executino space

Advantages of GPU page faulting:
- CUDA system doesn't need to sync all managed memory allocations to GPU before each kernel, faulting causes automatic migration
- OR page mapped to GPU addess space

### Performance tuning guidelines
- Faults should be avoided: fault handling takes a while since it may include TLB invalidates, data migrations and page table updates
- Data should be local to the access processor to minimize memory access latency and maximize bandwidth
- Overhead of migration may exceed the benefits of locality if data is constantly migrated

Hence we can _not_ use UM, since the UM drives can't detect common access patterns and optimize around it. WHen access patterns are non-obvious, it needs some guidance

Data prefetching is moving data to a processor's main memory and creating the mapping the page tables BEFORE data processing begins; aims to avoid faults and establish locality.

```cpp
    cudaError_t cudaMemPrefetchAsync(
        const void *devPtr,     // memory region
        size_t count,           // number of bytes
        inst dstDevice,         // device ID
        cudaStream_t stream
    );
```