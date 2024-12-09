# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -std=c++17 -O2 -fopenmp
# Executable name
TARGET = project
# Source files
SOURCES = main.cpp Algorithms.cpp Graph.cpp
# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default arguments
N = 50
P = 0.3
W = 1
K = 3
T = 2
A = 3

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Run the program with default or user-provided arguments
run: $(TARGET)
	./$(TARGET) -n $(N) -p $(P) -w $(W) -k $(K) -t $(T) -a $(A)