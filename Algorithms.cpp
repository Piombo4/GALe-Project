#include <random>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include "Graph.cpp"
#include <omp.h>
class Algorithms
{
private:
    /** Generates a random value from an exponential distribution
     * @param beta a value between 0 and 1
     * @return the random value
     */
    static double generateExponential(double beta)
    {
        static std::default_random_engine generator(static_cast<unsigned>(std::time(0)));
        std::exponential_distribution<double> distribution(beta);
        return distribution(generator);
    }
    /** Finds boundary edges between clusters in a graph
     * @param clusters a set of clusters
     * @param G a graph
     * @return a list of edges that connect nodes from one cluster to nodes of a different cluster
     */
    static std::vector<std::pair<int, int>> getBoundaryEdges(const std::map<int, std::vector<int>> &clusters, Graph &G)
    {
        std::vector<std::pair<int, int>> boundaryEdges;

        auto adj = G.adjList();
        for (const auto &[center, nodes] : clusters)
        {
            std::unordered_set<int> clusterSet(nodes.begin(), nodes.end());
            for (auto v : nodes)
            {
                for (auto &[neigh, w] : adj[v])
                {
                    if (!clusterSet.count(neigh))
                    {

                        boundaryEdges.push_back({v, neigh});
                    }
                }
            }
        }

        return boundaryEdges;
    }
    /** Finds boundary edges between clusters in a graph.
     *  It's the parallelized version of getBoundaryEdges using OpenMP.
     * @param clusters a set of clusters
     * @param G a graph
     * @return a list of edges that connect nodes from one cluster to nodes of a different cluster
     */
    static std::vector<std::pair<int, int>> getBoundaryEdgesParallel(const std::map<int, std::vector<int>> &clusters, Graph &G, int nThreads)
    {
        auto adj = G.adjList();

        // Vector to store thread-local cluster data
        // this way we can avoid critical section and merge all the data at the end
        std::vector<std::pair<int, std::vector<int>>> clusterData(clusters.begin(), clusters.end());

        // Thread-local storage for boundary edges
        std::vector<std::vector<std::pair<int, int>>> edgesPerThread(nThreads);
        for (int i = 0; i < nThreads; i++)
        {
            edgesPerThread[i].reserve(clusterData.size() / nThreads + 1); // Pre-allocate thread-local memory
        }

#pragma omp parallel for schedule(dynamic) num_threads(nThreads)
        for (int i = 0; i < (int)clusterData.size(); i++)
        {
            int tid = omp_get_thread_num();
            auto &[center, nodes] = clusterData[i];
            std::unordered_set<int> clusterSet(nodes.begin(), nodes.end());

            for (auto v : nodes)
            {
                for (auto &[neigh, w] : adj.at(v))
                {
                    if (!clusterSet.count(neigh))
                    {
                        edgesPerThread[tid].emplace_back(v, neigh);
                    }
                }
            }
        }

        // Calculate total size for all edges
        size_t totalSize = 0;
        for (auto &vec : edgesPerThread)
        {
            totalSize += vec.size();
        }

        // Reserve space for the total number of edges
        std::vector<std::pair<int, int>> boundaryEdges;
        boundaryEdges.reserve(totalSize);

        // Merge edges from all threads into a single vector
        for (auto &vec : edgesPerThread)
        {
            boundaryEdges.insert(boundaryEdges.end(), std::make_move_iterator(vec.begin()), std::make_move_iterator(vec.end()));
            vec.clear();
            vec.shrink_to_fit(); // Release unused capacity
        }

        return boundaryEdges;
    }

public:
    /** Graph decomposition algorithm. It generates a partition of V into
     * subsets X1, · · · , Xk, and a center ci for each Xi. It also
     * outputs a spanning tree for each cluster rooted at its center.
     * @param graph - the graph we want to partition
     * @param beta - a value between 0 and 1
     * @return a map representing the clusters, the key is the center and the vector are the other nodes in the cluster.
     */
    static std::map<int, std::vector<int>> ESTCluster(Graph &graph, double beta)
    {

        std::map<int, double> delta;
        std::map<int, std::vector<int>> clusters;

        // 1) For each vertex u, pick δu independently from the exponential distribution exp(β).
        for (const auto &[node, _] : graph.adjList())
        {
            delta[node] = generateExponential(beta);
        }
        std::map<int, std::map<int, int>> allDistances; // Stores precomputed distances

        // Step 1: Precompute shortest paths for all nodes
        for (const auto &[u, _] : graph.adjList())
        {
            allDistances[u] = graph.shortestPath(u);
        }
        // 2) Each node v selects the center that minimizes (distance(u, v) - delta[u]).
        // The node (center) with the lowest adjusted distance is the "winner" and attracts v into its cluster.
        for (const auto &[v, _] : graph.adjList())
        {
            double minValue = std::numeric_limits<double>::infinity();
            int closestCenter = -1;

            for (const auto &[u, _] : graph.adjList())
            {

                double adjustedDistance = allDistances[u][v] - delta[u];
                if (adjustedDistance < minValue)
                {
                    minValue = adjustedDistance;
                    closestCenter = u;
                }
            }
            clusters[closestCenter].push_back(v);
        }

        return clusters;
    }
    /** Parallel version of the ESTCluster algorithm. It generates a partition of V into
     * subsets X1, · · · , Xk, and a center ci for each Xi. It also
     * outputs a spanning tree for each cluster rooted at its center.
     * @param graph the graph we want to partition
     * @param beta a value between 0 and 1
     * @return a map representing the clusters, the key is the center and the vector are the other nodes in the cluster.
     */
    static std::map<int, std::vector<int>> ParallelESTCluster(Graph &graph, double beta, int nThreads = 2)
    {

        // Step 1: Generate random delta values for each node
        std::map<int, double> delta;
        for (const auto &[node, _] : graph.adjList())
        {
            delta[node] = generateExponential(beta);
        }

        // Thread-local storage using a vector for each thread
        std::vector<std::vector<std::pair<int, int>>> localClusters(nThreads);
        std::vector<int> keys;

        // Extract the list of graph nodes
        for (const auto &[key, _] : graph.adjList())
        {
            keys.push_back(key);
        }
        std::map<int, std::map<int, int>> allDistances;
        std::vector<std::map<int, std::map<int, int>>> threadLocalDistances(nThreads);

        // We create a parallel region
#pragma omp parallel num_threads(nThreads)
        {
            int tid = omp_get_thread_num(); // Thread ID
            std::map<int, std::map<int, int>> &localDistances = threadLocalDistances[tid];
            // If placed in a parallel region, "#pragma omp for" takes a loop and divides its iterations among
            // the threads that are currently executing in a parallel region, schedule(dynamic) divides the work evenly between threads
#pragma omp for schedule(dynamic)
            for (size_t i = 0; i < keys.size(); i++)
            {
                int u = keys[i];

                localDistances[u] = graph.shortestPath(u);
            }
        }

        // Merge thread-local distances into a single global map
        for (int t = 0; t < nThreads; t++)
        {
            for (const auto &[node, distances] : threadLocalDistances[t])
            {
                allDistances[node] = distances;
            }
        }

        // We create a parallel region
#pragma omp parallel num_threads(nThreads)
        {
            int tid = omp_get_thread_num(); // Thread ID
            auto &localEdges = localClusters[tid];
            // If placed in a parallel region, "#pragma omp for" takes a loop and divides its iterations among
            // the threads that are currently executing in a parallel region
#pragma omp for
            for (size_t i = 0; i < keys.size(); i++)
            {
                int v = keys[i];
                double minValue = std::numeric_limits<double>::infinity();
                int closestCenter = -1;

                // Find the closest cluster center for node v
                for (const auto &[u, _] : graph.adjList())
                {
                    double adjustedDistance = allDistances[u][v] - delta[u];
                    if (adjustedDistance < minValue)
                    {
                        minValue = adjustedDistance;
                        closestCenter = u;
                    }
                }

                // Add the (center, member) pair for this node to the local thread's storage
                localEdges.push_back({closestCenter, v});
            }
        }

        // Merge the local thread results into the global structure
        std::map<int, std::vector<int>> clusters;

        for (int t = 0; t < nThreads; t++)
        {
            for (const auto &[center, member] : localClusters[t])
            {
                clusters[center].push_back(member);
            }
        }

        return clusters;
    }

    /** Parallel version of the ESTCluster algorithm using a super source as a virtual node.
     *  WORK IN PROGRESS
     * @param graph the graph we want to partition
     * @param beta a value between 0 and 1
     * @return a map representing the clusters, the key is the center and the vector are the other nodes in the cluster.
     */

    static std::map<int, std::vector<int>> ParallelESTClusterWithSuperSource(Graph &graph, double beta, int nThreads = 2)
    {

        const int super_source = -1;
        std::unordered_map<int, double> delta;
        std::vector<std::pair<int, double>> superSourceEdges;
        auto adj = graph.adjList();
        std::vector<int> keys;

        for (const auto &[key, _] : graph.adjList())
            keys.push_back(key);

        // Thread local structures
        std::vector<std::vector<std::pair<int, double>>> threadSuperSourceEdges(nThreads);

        // parallel delta computing
#pragma omp parallel for num_threads(nThreads)
        for (size_t i = 0; i < keys.size(); i++)
        {
            int u = keys[i];
            double deltaVal = generateExponential(beta);
            delta[u] = deltaVal;
            threadSuperSourceEdges[omp_get_thread_num()].emplace_back(u, deltaVal);
        }

        // Merge super-source edges from all threads
        for (const auto &edges : threadSuperSourceEdges)
        {
            superSourceEdges.insert(superSourceEdges.end(), edges.begin(), edges.end());
        }

        // Add super-source edges to the graph
        for (const auto &[u, weight] : superSourceEdges)
        {
            adj[super_source].emplace(u, weight);
        }

        // Step 2: Shortest path calculation with thread-local distances.
        std::vector<std::unordered_map<int, double>> threadDistances(nThreads);
        // Each map contains pairs of (node, parent_node), representing the parent of node in the shortest path tree.
        std::vector<std::unordered_map<int, int>> threadParents(nThreads);
        std::vector<int> frontier = {super_source};

        while (!frontier.empty())
        {
            std::vector<int> nextFrontier;

            // Each thread has its own local copy of the next frontier
            std::vector<std::vector<int>> threadNextFrontiers(nThreads);

#pragma omp parallel for num_threads(nThreads)
            for (size_t i = 0; i < frontier.size(); i++)
            {
                // The loop processes each node u from the frontier in parallel
                int u = frontier[i];
                int tid = omp_get_thread_num();

                for (const auto &[v, weight] : adj[u])
                {
                    // We calculate the distance to v from u by adding the edge weight to the distance to u.
                    double newDist = threadDistances[tid][u] + weight;

                    // If the node v has not been visited before
                    //  or if the new distance is shorter than the previously known distance, we update the distance.
                    if (threadDistances[tid].find(v) == threadDistances[tid].end() || newDist < threadDistances[tid][v])
                    {
                        threadDistances[tid][v] = newDist;
                        threadParents[tid][v] = u;
                        threadNextFrontiers[tid].push_back(v);
                    }
                }
            }
            // All thread-local nextFrontiers are merged into a single global nextFrontier.
            for (auto &localFrontier : threadNextFrontiers)
            {
                nextFrontier.insert(nextFrontier.end(), localFrontier.begin(), localFrontier.end());
                localFrontier.clear();
            }
            // nextFrontier becomes the new set of nodes to process in the next iteration.
            frontier = nextFrontier;
        }

        // We merge parent information from all threads in a global map
        std::unordered_map<int, int> parent;
        for (const auto &localParent : threadParents)
        {
            for (const auto &[node, parent_node] : localParent)
            {
                parent[node] = parent_node;
            }
        }

        // Each thread keeps its own cluster map to avoid race conditions
        std::vector<std::map<int, std::vector<int>>> threadClusters(nThreads);
        std::map<int, std::vector<int>> clusters;

#pragma omp parallel for num_threads(nThreads)
        for (size_t i = 0; i < keys.size(); i++)
        {
            int node = keys[i];
            int parent_node = parent[node];
            int tid = omp_get_thread_num();
            // If the parent of node is the super_source, the node is its own cluster center.
            // Otherwise, the node belongs to the cluster of its parent.
            if (parent_node == super_source)
            {
                threadClusters[tid][node].push_back(node);
            }
            else
            {
                threadClusters[tid][parent_node].push_back(node);
            }
        }

        // Merge thread-local clusters into the final clusters
        for (auto &localCluster : threadClusters)
        {
            for (const auto &[center, members] : localCluster)
            {
                clusters[center].insert(clusters[center].end(), members.begin(), members.end());
            }
        }

        return clusters;
    }
    /** Spanner construction for unweighted graphs.
     *  @param G a graph
     *  @param k value > 1. It represents the stretch factor
     *  @return The O(k) spanner of G
     */
    static Graph UnweightedSpanner(Graph &G, double k)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = (int)G.adjList().size();
        double beta = std::log(n) / (2.0 * k);
        // Compute EST clustering
        auto clusters = ESTCluster(G, beta);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "-    Sequential ESTCluster execution time:  " << duration.count() << " microseconds" << std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();

        Graph H(G.nVertices());

        // Add edges connecting cluster centers to their respective cluster nodes
        for (auto &[c, nodes] : clusters)
        {
            // c è il centro, nodes sono i nodi del cluster
            // Aggiungi un arco tra il centro e ogni altro nodo del cluster (se non coincidente)
            for (auto v : nodes)
            {
                // Only add edges between the cluster center and its other nodes
                if (v != c)
                {
                    H.addEdge(c, v, 1);
                }
            }
        }

        // We identify boundary edges and add them to H
        auto boundaryEdges = getBoundaryEdges(clusters, G);
        for (auto &edge : boundaryEdges)
        {
            H.addEdge(edge.first, edge.second, 1);
        }
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
        std::cout << "-    Sequential UnweightedSpanner execution time: " << duration2.count() << " microseconds" << std::endl;
        std::cout << "-    Total execution time: " << duration.count() + duration2.count() << " microseconds" << std::endl;

        return H;
    }
    /** Parallel version of UnweightedSpanner, for spanner construction for unweighted graphs.
     *  @param G a graph
     *  @param k value > 1. It represents the stretch factor
     *  @param nThreads the number of threads used for parallelization
     *  @return The O(k) spanner of G
     */
    static Graph ParallelUnweightedSpanner(Graph &G, double k, int nThreads, bool flag)
    {
        auto start = std::chrono::high_resolution_clock::now();
        int n = G.nVertices();
        double beta = std::log(n) / (2.0 * k);

        // Computing the clusters with parallel ESTCluster
        //NOT USED AT THE MOMENT
        // auto clusters = flag ? ParallelESTClusterWithSuperSource(G, beta, nThreads) : ParallelESTCluster(G, beta, nThreads);
        auto clusters = ParallelESTCluster(G, beta, nThreads);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        std::cout << (flag ? "-    ParallelESTClusterWithSuperSource execution time: " : "-    ParallelESTCluster execution time: ") << duration.count() << " microseconds" << std::endl;
        auto start2 = std::chrono::high_resolution_clock::now();
        Graph H(n);

        // For each cluster, we connect the cluster center c to each node in the cluster.
        for (auto &[c, nodes] : clusters)
        {
            for (auto v : nodes)
            {
                if (v != c)
                {
                    H.addEdge(c, v, 1.0);
                }
            }
        }

        // We identify the boundary edges
        auto boundaryEdges = getBoundaryEdgesParallel(clusters, G, nThreads);
        // We add them to the Spanner
        for (auto &edge : boundaryEdges)
        {
            H.addEdge(edge.first, edge.second, 1.0);
        }
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
        std::cout << "-    Parallel UnweightedSpanner execution time: " << duration2.count() << " microseconds" << std::endl;
        std::cout << "-    Total execution time: " << duration.count() + duration2.count() << " microseconds" << std::endl;
        return H;
    }
    /** Generates a random graph using Erdős–Rényi model
     * @param nVertices the number of vertices in the graph
     * @param p the probability of an edge
     * @param maxWeight the max weight an edge can have
     * @return a Graph
     */
    static Graph generateRandomGraph(int nVertices, double p, double maxWeight)
    {
        Graph graph = Graph(nVertices);
        std::default_random_engine generator(static_cast<unsigned>(std::time(0)));
        std::uniform_real_distribution<double> probabilityDist(0.0, 1.0);
        std::uniform_real_distribution<double> weightDist(1.0, maxWeight);

        for (int u = 0; u < nVertices; ++u)
        {
            for (int v = u + 1; v < nVertices; ++v)
            {
                if (probabilityDist(generator) < p)
                {
                    double weight = weightDist(generator);
                    graph.addEdge(u, v, weight);
                }
            }
        }

        return graph;
    }
};