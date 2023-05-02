#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1>   dim2;
typedef std::vector<dim2>   dim3;

void progress_bar(double progress, double total)
{
    /*
    progress bar
    */
    int barWidth = 70;
    std::cout << "[";
    int pos = barWidth * progress / total;
    for (int i = 0; i < barWidth; ++i)
    {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress / total * 100.0) << " %\r";
    std::cout.flush();
    if (progress == total)
        std::cout << std::endl;
}

double randn(double min=0, double max=1)
{
    /*
    random number generator
    */
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

  std::vector<double> linspaced;

  double start = static_cast<double>(start_in);
  double end = static_cast<double>(end_in);
  double num = static_cast<double>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

template<typename T>
std::vector<double> logspace(double start, double end, int num)
{
    /*
    logspace
    */
    std::vector<double> result;
    double delta = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i)
    {
        result.push_back(pow(10, start + delta * i));
    }
    return result;
}
