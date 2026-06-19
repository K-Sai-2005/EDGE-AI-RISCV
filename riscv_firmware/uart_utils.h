/**
 * uart_utils.h
 * Minimal UART print utilities — no printf dependency
 * Works in simulation (semihosting) and on real FE310 hardware
 */

#ifndef UART_UTILS_H
#define UART_UTILS_H

#include <stdint.h>

#ifdef SIMULATION_MODE
  /* In Freedom Studio simulation, route to semihosted printf */
  #include <stdio.h>
  #define uart_print(s)       printf("%s", s)
  #define uart_print_uint(n)  printf("%lu", (unsigned long)(n))
  #define uart_print_int(n)   printf("%ld", (long)(n))
#else
  /* Bare-metal UART0 on FE310 */
  void uart_print(const char *s);
  void uart_print_uint(uint32_t n);
  void uart_print_int(int32_t n);
#endif

#endif /* UART_UTILS_H */
