// [[kernel]]
// void add(
//   constant uint *inA [[buffer(0)]],
//   constant uint *inB [[buffer(1)]],
//   device uint *result [[buffer(2)]],
//   uint index [[thread_position_in_grid]])
// {
//   result[index] = inA[index] + inB[index];
// }
#include <metal_stdlib>
using namespace metal;
// [[kernel]]
kernel void add(
  texture2d<uint, access::read> A [[texture(0)]],
  texture2d<uint, access::read> B [[texture(1)]],
  texture2d<uint, access::write> C [[texture(2)]],
  uint2 gid [[thread_position_in_grid]])
{
  C.write(A.read(gid) + B.read(gid), gid);
}

