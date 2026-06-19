/**
 * main.c
 * Concrete Crack Detection — Bare-Metal Edge AI Inference
 * Target : SiFive FE310-G002 (RV32IMAC) on VSDSquadron PRO
 * Toolchain: RISC-V GNU (riscv64-unknown-elf-gcc)
 *
 * What this does:
 *   1. Loads a pre-baked INT8 test image from test_image.h
 *   2. Runs a lightweight dense inference engine using integer arithmetic
 *   3. Prints the result to UART (visible in Freedom Studio console)
 *
 * NOTE: Full TFLite flatbuffer parsing requires TFLite Micro library.
 *       This file implements a standalone integer-arithmetic inference
 *       engine for the exported quantized weights (dense layers only),
 *       which is the approach suitable for bare-metal without an OS.
 *       For full TFLite Micro support, see README Section 4.
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "model_params.h"
#include "inference_engine.h"
#include "test_image.h"
#include "uart_utils.h"

/* ─── Board-level UART init (FE310 UART0) ──────────────────────────────── */
#ifndef SIMULATION_MODE
  #include "platform.h"   /* Freedom Metal / SiFive SDK */
  #define uart_init()     metal_uart_init(metal_uart_get_device(0), 115200)
#else
  /* In simulation (Freedom Studio GDB), printf routes to semihosting */
  #define uart_init()     /* no-op */
#endif

/* ─── Simple LED blink to signal result (GPIO 19 = onboard LED FE310) ─── */
#define LED_PIN  19

static void led_signal(int crack_detected) {
#ifndef SIMULATION_MODE
    /* Toggle LED fast for crack, slow for no crack */
    volatile uint32_t *gpio_output = (volatile uint32_t *)0x10012008;
    int blinks = crack_detected ? 5 : 2;
    for (int i = 0; i < blinks; i++) {
        *gpio_output |=  (1u << LED_PIN);
        for (volatile int d = 0; d < 200000; d++);
        *gpio_output &= ~(1u << LED_PIN);
        for (volatile int d = 0; d < 200000; d++);
    }
#else
    (void)crack_detected;
#endif
}

/* ─── Main ─────────────────────────────────────────────────────────────── */
int main(void) {
    uart_init();

    uart_print("\r\n");
    uart_print("================================================\r\n");
    uart_print("  Concrete Crack Detection — Edge AI (RISC-V)  \r\n");
    uart_print("  SiFive FE310-G002 | VSDSquadron PRO           \r\n");
    uart_print("================================================\r\n\r\n");

    uart_print("[INFO] Model size       : ");
    uart_print_uint(MODEL_DATA_LEN);
    uart_print(" bytes\r\n");

    uart_print("[INFO] Input dimensions : ");
    uart_print_uint(IMG_SIZE);
    uart_print("x");
    uart_print_uint(IMG_SIZE);
    uart_print("x");
    uart_print_uint(IMG_CHANNELS);
    uart_print("\r\n");

    uart_print("[INFO] Quantization     : INT8\r\n\r\n");

    /* ── Run inference ─────────────────────────────────────────────────── */
    uart_print("[INFO] Loading test image...\r\n");

    int8_t output_score = 0;
    InferenceStatus status = run_inference(
        g_test_image,          /* INT8 input tensor  */
        IMG_SIZE * IMG_SIZE * IMG_CHANNELS,
        g_model_data,          /* quantized flatbuffer */
        MODEL_DATA_LEN,
        &output_score          /* raw INT8 output score */
    );

    if (status != INFERENCE_OK) {
        uart_print("[ERROR] Inference failed! Status: ");
        uart_print_uint((uint32_t)status);
        uart_print("\r\n");
        return -1;
    }

    /* ── Decode result ─────────────────────────────────────────────────── */
    /*
     * INT8 output from sigmoid layer:
     *   > CRACK_THRESHOLD_INT8 (0)  →  Crack detected
     *   <= CRACK_THRESHOLD_INT8 (0) →  No crack
     */
    int crack_detected = (output_score > CRACK_THRESHOLD_INT8) ? 1 : 0;

    uart_print("[INFO] Raw INT8 output score : ");
    uart_print_int((int32_t)output_score);
    uart_print("\r\n");

    uart_print("\r\n");
    uart_print("┌─────────────────────────────────┐\r\n");
    if (crack_detected) {
        uart_print("│  RESULT: ⚠ CRACK DETECTED       │\r\n");
    } else {
        uart_print("│  RESULT: ✓ NO CRACK — SURFACE OK │\r\n");
    }
    uart_print("└─────────────────────────────────┘\r\n");

    uart_print("\r\n[INFO] True label from test_image.h: ");
    uart_print(TEST_IMG_TRUE_LABEL == 1 ? "CRACK" : "NO_CRACK");
    uart_print("\r\n");

    if (crack_detected == TEST_IMG_TRUE_LABEL) {
        uart_print("[PASS] ✓ Prediction matches ground truth!\r\n");
    } else {
        uart_print("[FAIL] ✗ Mismatch — check quantization calibration.\r\n");
    }

    uart_print("\r\n[INFO] Signalling result via LED...\r\n");
    led_signal(crack_detected);

    uart_print("[INFO] Inference complete. Halting core.\r\n");

    /* Halt — spin forever (bare-metal, no OS) */
    while (1) { __asm__ volatile("wfi"); }

    return 0;
}
