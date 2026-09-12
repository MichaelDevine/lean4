/* Keep CPU and real-GPU fixtures identical, without changing existing tests. */
#define LEAN_REDUCTION_NATURAL_DEVICE_CONTROL
#include "natural.cpp"
int main() { return natural_controls(true); }
