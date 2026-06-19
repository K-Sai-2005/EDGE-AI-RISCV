/**
 * uart_utils.c
 * Bare-metal UART0 implementation for SiFive FE310-G002
 * MMIO register addresses from FE310-G002 Manual v19p05
 */

#ifndef SIMULATION_MODE

#include "uart_utils.h"

/* FE310 UART0 base address */
#define UART0_BASE       0x10013000UL
#define UART_TXDATA      (*(volatile uint32_t *)(UART0_BASE + 0x00))
#define UART_RXDATA      (*(volatile uint32_t *)(UART0_BASE + 0x04))
#define UART_TXCTRL      (*(volatile uint32_t *)(UART0_BASE + 0x08))
#define UART_DIV         (*(volatile uint32_t *)(UART0_BASE + 0x18))

/* Baud rate: CPU clock 16 MHz, target 115200 bps → div = 16000000/115200 - 1 = 137 */
#define UART_DIV_115200  137U

static void uart_putc(char c) {
    /* Wait until TX FIFO is not full (bit 31 = full) */
    while (UART_TXDATA & 0x80000000UL);
    UART_TXDATA = (uint32_t)(unsigned char)c;
}

void uart_print(const char *s) {
    while (*s) uart_putc(*s++);
}

void uart_print_uint(uint32_t n) {
    char buf[12];
    int  i = 10;
    buf[11] = '\0';
    if (n == 0) { uart_putc('0'); return; }
    while (n && i >= 0) {
        buf[i--] = '0' + (n % 10);
        n /= 10;
    }
    uart_print(&buf[i + 1]);
}

void uart_print_int(int32_t n) {
    if (n < 0) { uart_putc('-'); uart_print_uint((uint32_t)(-n)); }
    else        { uart_print_uint((uint32_t)n); }
}

#endif /* !SIMULATION_MODE */
