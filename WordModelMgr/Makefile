# Adapted from http://make.mad-scientist.net/papers/advanced-auto-dependency-generation/.

# Initialize variables.
CXXFLAGS := -fPIC
SYMBOLS :=
LIB_NAME := libwordmodelmgr.so
SRCDIR := src
DEBUG_OBJDIR := obj/debug
RELEASE_OBJDIR := obj/release
LIBDIR := lib


# Define top-level rules.
.PHONY: all
all:
	@echo "Please specify 'debug' or 'release'."

.PHONY: debug
debug: CXXFLAGS += -g
debug: DEPFLAGS = -MMD -MP -MF $(DEBUG_OBJDIR)/$*.Td
debug: OBJDIR = $(DEBUG_OBJDIR)
debug: library_debug

.PHONY: release
release: CXXFLAGS += -O2
release: DEPFLAGS = -MMD -MP -MF $(RELEASE_OBJDIR)/$*.Td
release: OBJDIR = $(RELEASE_OBJDIR)
release: library_release


# Define build rules.
# Configure automatic dependency generation as side-effect of compilation.
COMPILE.cpp = $(CXX) $(CXXFLAGS) $(DEPFLAGS) $(SYMBOLS) -c
POSTCOMPILE = mv -f $(OBJDIR)/$*.Td $(OBJDIR)/$*.d

$(DEBUG_OBJDIR)/%.o: src/%.cpp
$(DEBUG_OBJDIR)/%.o: src/%.cpp $(DEBUG_OBJDIR)/%.d
	$(COMPILE.cpp) $< -o $@
	$(POSTCOMPILE)

$(DEBUG_OBJDIR)/%.d: ;
.PRECIOUS: $(DEBUG_OBJDIR)/%.d

$(RELEASE_OBJDIR)/%.o: src/%.cpp
$(RELEASE_OBJDIR)/%.o: src/%.cpp $(RELEASE_OBJDIR)/%.d
	$(COMPILE.cpp) $< -o $@
	$(POSTCOMPILE)

$(RELEASE_OBJDIR)/%.d: ;
.PRECIOUS: $(RELEASE_OBJDIR)/%.d


# Define sources and objects.
SOURCES := $(wildcard src/*.cpp)
DEBUG_OBJECTS = $(patsubst src/%,$(DEBUG_OBJDIR)/%,$(SOURCES:.cpp=.o))
RELEASE_OBJECTS = $(patsubst src/%,$(RELEASE_OBJDIR)/%,$(SOURCES:.cpp=.o))

# Define rules.
.PHONY: library_debug
library_debug: make_directories $(DEBUG_OBJECTS)
	@echo "[Building debug library]"
	$(CXX) -shared -Wl,-soname,$(LIB_NAME) -o $(LIBDIR)/$(LIB_NAME) $(DEBUG_OBJECTS)

.PHONY: library_release
library_release: make_directories $(RELEASE_OBJECTS)
	@echo "[Building release library]"
	$(CXX) -shared -Wl,-soname,$(LIB_NAME) -o $(LIBDIR)/$(LIB_NAME) $(RELEASE_OBJECTS)

.PHONY: make_directories
make_directories:
	@echo "[Creating directories]"
	@mkdir -p $(LIBDIR)
	@mkdir -p $(OBJDIR)

.PHONY: clean
clean:
	@echo "[Cleaning debug and release]"
	@rm -f $(DEBUG_OBJDIR)/*.o
	@rm -f $(DEBUG_OBJDIR)/*.d
	@rm -f $(RELEASE_OBJDIR)/*.o
	@rm -f $(RELEASE_OBJDIR)/*.d
	@rm -f $(LIBDIR)/$(LIB_NAME)


# Include existing dependency files - keep at end of makefile.	
-include $(patsubst src/%,$(DEBUG_OBJDIR)/%.d,$(basename $(SOURCES)))
-include $(patsubst src/%,$(RELEASE_OBJDIR)/%.d,$(basename $(SOURCES)))

