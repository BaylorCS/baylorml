
INCLUDE_DIR := include
SRC_DIR     := source
OBJ_DIR     := objects
TESTS_DIR   := tests

STATIC_LIB_NAME := libml.a

CC := $(TARGET)g++
AR := $(TARGET)ar
CC_FLAGS_LOCAL := $(CC_FLAGS) \
	-g -O2 -fvisibility=hidden -Wall -Wextra -Werror -pedantic \
	-Wswitch-default -Wcast-qual -Wcast-align -Wconversion \
	-Wno-unused-parameter -Wno-long-long -Wno-sign-conversion \
	-D_FILE_OFFSET_BITS=64 \
	-I ../librho/include \
	-I $(INCLUDE_DIR)  # consider: -Wold-style-cast -Wshadow

ifeq ($(shell uname),Linux)
	# Linux stuff:
	CC_FLAGS_LOCAL += -rdynamic -Wdouble-promotion
else
ifeq ($(shell uname),Darwin)
	# OSX stuff:
	CC_FLAGS_LOCAL +=
else
	# Mingw and Cygwin stuff:
endif
endif

CPP_SRC_FILES = $(shell find $(SRC_DIR) -name '*.cpp' -type f)
CPP_OBJ_FILES = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SRC_FILES))

all : $(CPP_OBJ_FILES)

install : ensure_root uninstall all
	@echo
	@cp -r $(INCLUDE_DIR)/* /usr/local/include
	@cp $(OBJ_DIR)/$(STATIC_LIB_NAME) /usr/local/lib
	@echo "Install successful."

uninstall : ensure_root
	@(cd $(INCLUDE_DIR) && for i in *; do rm -rf /usr/local/include/$$i; done)

ensure_root :
	$(if $(shell whoami | grep root),,\
	@echo 'You must be root to run to perform that operation.' && exit 1; \
	)

test : all
	@$(TESTS_DIR)/RunTests.bash

clean :
	@rm -rf $(OBJ_DIR)
	@echo "Clean successful."

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
	@echo "Compiling $< ..."
	@mkdir -p $(@D)
	$(CC) $(CC_FLAGS_LOCAL) -c -o $@ $<
	$(AR) crsv $(OBJ_DIR)/$(STATIC_LIB_NAME) $@
	@echo

.PHONY : all install uninstall ensure_root test clean
