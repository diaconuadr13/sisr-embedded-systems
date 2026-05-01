CC      = gcc
CFLAGS  = -O2 -std=c99 -Wall -Wextra -Icsrc/include -lm

SRCS    = csrc/sisr_primitives.c csrc/sisr_weights.c \
          csrc/espcn_light.c csrc/carn_m.c

ESP32_CC     = xtensa-esp32-elf-gcc
ESP32_CFLAGS = -O2 -std=c99 -mlongcalls -Icsrc/include

.PHONY: test esp32 clean

test: $(SRCS) tests/main_test.c
	$(CC) $(CFLAGS) $^ -o sisr_test -lm

esp32: $(SRCS) esp32/main.c
	$(ESP32_CC) $(ESP32_CFLAGS) $^ -o sisr_esp32.elf -lm

clean:
	rm -f sisr_test sisr_esp32.elf
