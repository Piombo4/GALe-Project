# Graph Algorithms Project
## Introduction 
This repo presents the implementation of a graph decomposition routine called
“ESTCluser” and its use for “UnweightedSpanner”, an algorithm to build
spanners for unweighted graphs.
For both of the algorithms, a sequential and a parallel version were implemented.
For the ESTCluster a signicant speedup was noticed. For UnweightedSpanner there were
improvements, but not as signicant as the previous one.
## Results
<figure>
   <img src="/images/sparse.png" width="450" height="100">
    <figcaption>Sparse graph</figcaption>
</figure>
</br>
<figure>
   <img src="/images/dense.png" width="450" height="100">
    <figcaption>Dense graph</figcaption>
</figure>

All the values are in microseconds
## Usage 
The code was originally run on Windows 11 by firstly building the executable using:</br></br>
`g++ -fopenmp main.cpp -o program`</br></br>
And running it by calling:</br></br>
`./programName -n <num_vertices> -p <edge_probability> -w <max_weight> -k
<stretch_factor> -t <n_threads> -a <execution_type: 1..5>`</br></br>
By using the Makefile provided it’s possible to build the program and execute the same in
the same way as shown above. By default, the executable will be called “project”

