#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <fstream>
#include <cmath>

class Graph
{
private:
    int _nVertices;
    std::map<int, std::set<std::pair<int, double>>> _adjList;

public:
    Graph(int vertices) : _nVertices(vertices) {}
    ~Graph() {}

    std::map<int, std::set<std::pair<int, double>>> adjList() { return _adjList; }
    int nVertices() { return _nVertices; }
    /** Calculates the number of edges in the graph
     * @return the number of edges in the graph
     */
    int getTotalEdges()
    {
        int totalEdges = 0;
        for (const auto &[node, neighbors] : _adjList)
        {
            totalEdges += neighbors.size();
        }
        return totalEdges / 2;
    }
    /** Adds an edge between u and v, since it's an undirected graph we add the edge "twice"
     * @param u - first vertex
     * @param v - second vertex
     * @param weight - the weight of the edge, if all edges have weight 1, the graph can be considered unweighted
     */
    void addEdge(int u, int v, double weight)
    {
        _adjList[u].emplace(v, weight);
        _adjList[v].emplace(u, weight);
    }
    /**
     * Prints the graph
     */
    void printGraph() const
    {
        for (const auto &[node, neighbors] : _adjList)
        {
            std::cout << "Node " << node << ": ";
            for (const auto &[neighbor, weight] : neighbors)
            {
                std::cout << "(" << neighbor << ", " << weight << ") ";
            }
            std::cout << "\n";
        }
    }

    /** BFS algorithm to calculate the shortest path from a source to all the other nodes in the graph
     * @param source - the source vertex
     * @return a map where each key is a node and the value is the minimum distance from the source
     */
    std::map<int, int> shortestPath(int source)
    {
        const int INF = std::numeric_limits<int>::max();
        std::map<int, int> distances;

        // Initialization
        for (const auto &[node, _] : _adjList)
        {
            distances[node] = INF;
        }
        distances[source] = 0;

        std::queue<int> q;
        q.push(source);

        while (!q.empty())
        {
            int u = q.front();
            q.pop();

            // Traverse all neighbors of the current node
            for (const auto &[v, _] : _adjList.at(u))
            {
                // If we found a shorter path to v, we update the distance and enqueue it
                if (distances[u] + 1 < distances[v])
                {
                    distances[v] = distances[u] + 1;
                    q.push(v);
                }
            }
        }

        return distances;
    }
    /** Export the graph to CSV
     *  @param filename - the name of the .csv file
     */

    void exportToCSV(const std::string &filename) const
    {
        std::ofstream file(filename);
        if (!file)
        {
            std::cerr << "Error: Unable to open file for writing.\n";
            return;
        }

        file << "Source,Target,Weight\n";
        for (const auto &[node, neighbors] : _adjList)
        {
            for (const auto &[neighbor, weight] : neighbors)
            {
                if (node < neighbor)
                {
                    file << node << "," << neighbor << "," << weight << "\n";
                }
            }
        }

        file.close();
        std::cout << "Graph exported to " << filename << " in CSV format.\n";
    }

    /** Export the graph to DOT for visualization using Graphviz
     *  @param filename - the name of the .csv file
     */
    void exportToDOT(const std::string &filename) const
    {
        std::ofstream file(filename);
        if (!file)
        {
            std::cerr << "Error: Unable to open file for writing.\n";
            return;
        }

        file << "graph G {\n";
        for (const auto &[node, neighbors] : _adjList)
        {
            for (const auto &[neighbor, weight] : neighbors)
            {
                if (node < neighbor)
                {
                    file << "    " << node << " -- " << neighbor << " [label=\"" << weight << "\"];\n";
                }
            }
        }
        file << "}\n";

        file.close();
        std::cout << "Graph exported to " << filename << " in DOT format.\n";
    }
};