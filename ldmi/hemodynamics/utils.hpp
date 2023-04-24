#include <vector>
#include <string>
#include <assert.h>
#include <iostream>

using std::string;
using std::vector;

typedef std::vector<double> dim1;
typedef std::vector<dim1>   dim2;
typedef std::vector<dim2>   dim3;

dim1 dot_2D1D(dim2 &A, dim1 &v)
{
    /*
    dot product of a 2D vector and a 1D vector, returns a 1D vector
    */
    assert(A.size() == v.size());
    assert(A[0].size() == v.size());
    dim1 result;
    result.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i)
    {
        result[i] = 0;
        for (size_t j = 0; j < A[0].size(); ++j)
        {
            result[i] += A[i][j] * v[j];
        }
    }

    return result;
}

double dot_1D0D(dim1 &a, double &d)
{
    /*
    dot product of a 1D vector and a double, returns a double
    */
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * d;
    }

    return result;
}

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