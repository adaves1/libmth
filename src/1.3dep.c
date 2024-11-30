#include <windows.h>

__delspec(dllexport) void elkonaur_sequence(m, range) {
  c = 4;
  while m >= range {
    c = m;
    m = c + m;
    c = m + c;
    return 0;
}
