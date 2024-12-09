
#include <random>
#include <ctime>
#include "Algorithms.cpp"

void printUsage(const char *programName)
{
    std::cout << "Usage: ./programName  -n <num_vertices> -p <edge_probability> -w <max_weight> -k <stretch_factor> -t <n_threads> -a <execution_type: 1..5>\n";
    std::cout << "flag -a: \n 1: SquentialUnweightedSpanner will be executed \n 2: ParallelUnweightedSpanner will be executed \n 3: Both of them will be executed\n 4: Sequential UnweightedSpanner will be run 30 times 5: Parallel UnweightedSpanner will be run 30 times \n";
    std::cout << "Example: " << programName << " -n 50 -p 0.3 -w 1 -t 2 -a 3\n";
}

int main(int argc, char *argv[])
{
    // Default values
    int nVertices = 50;
    double edgeProbability = 0.3;
    double maxWeight = 1;
    int nThreads = 2;
    int k = 3;
    int type = 4;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if ((arg == "-n" || arg == "--vertices") && i + 1 < argc)
        {
            nVertices = std::atoi(argv[++i]);
        }
        else if ((arg == "-p" || arg == "--probability") && i + 1 < argc)
        {
            edgeProbability = std::atof(argv[++i]);
        }
        else if ((arg == "-w" || arg == "--weight") && i + 1 < argc)
        {
            maxWeight = std::atof(argv[++i]);
        }
        else if ((arg == "-k" || arg == "--stretch") && i + 1 < argc)
        {
            k = std::atof(argv[++i]);
        }
        else if ((arg == "-t" || arg == "--threads") && i + 1 < argc)
        {
            nThreads = std::atoi(argv[++i]);
        }
        else if ((arg == "-a" || arg == "--exec") && i + 1 < argc)
        {
            type = std::atoi(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help")
        {
            printUsage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    std::cout << "Using the following parameters:\n";
    std::cout << "Number of vertices: " << nVertices << std::endl;
    std::cout << "Edge probability: " << edgeProbability << std::endl;
    std::cout << "Maximum edge weight: " << maxWeight << std::endl;
    std::cout << "Stretch factor: " << k << std::endl;
    std::cout << "Number of threads: " << nThreads << std::endl;
    auto g = Algorithms::generateRandomGraph(nVertices, edgeProbability, maxWeight);
    std::cout << "Graph generated, number of edges: " << g.getTotalEdges() << std::endl;
    switch (type)
    {
    case 1:
    {
        std::cout << "Running sequential version of unweighted spanner..." << std::endl;
        auto spanner = Algorithms::UnweightedSpanner(g, k);
        std::cout << "Spanner generated, number of edges: " << spanner.getTotalEdges() << std::endl;
        break;
    }
    case 2:
    {
        std::cout << "Running parallel version of unweighted spanner with " << nThreads << " threads" << std::endl;
        auto spanner2 = Algorithms::ParallelUnweightedSpanner(g, k, nThreads, false);
        std::cout << "Spanner generated, number of edges: " << spanner2.getTotalEdges() << std::endl;
        break;
    }

    case 3:
    {
        std::cout << "Running sequential version of unweighted spanner..." << std::endl;
        auto spanner = Algorithms::UnweightedSpanner(g, k);
        std::cout << "Spanner generated, number of edges: " << spanner.getTotalEdges() << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "Running parallel version of unweighted spanner with " << nThreads << " threads" << std::endl;
        auto spanner2 = Algorithms::ParallelUnweightedSpanner(g, k, nThreads, false);
        std::cout << "Spanner generated, number of edges: " << spanner2.getTotalEdges() << std::endl;
        break;
    }

    case 4:
    {
        std::cout << "Running 30 cycles of Sequential UnweightedSpanners..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        for (int i = 0; i < 30; i++)
        {
            auto spanner2 = Algorithms::UnweightedSpanner(g, k);
            std::cout << "Spanner generated, number of edges: " << spanner2.getTotalEdges() << std::endl;
        }
        std::cout << "DONE!" << std::endl;

        break;
    }
    case 5:
    {
        std::cout << "Running 30 cycles of Parallel UnweightedSpanners..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        for (int i = 0; i < 30; i++)
        {
            auto spanner2 = Algorithms::ParallelUnweightedSpanner(g, k, nThreads, false);
            std::cout << "Spanner generated, number of edges: " << spanner2.getTotalEdges() << std::endl;
        }
        std::cout << "DONE!" << std::endl;

        break;
    }
    default:
        break;
    }
}