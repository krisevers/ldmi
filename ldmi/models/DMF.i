
%module DMF

%{
#include "DMF.hpp"
%}

%include stl.i
%include "std_string.i"
/* instantiate the required template specializations */
namespace std {
        %template(IntVector)            vector<int>;
        %template(DoubleVector)         vector<double>;
        %template(DoubleVector2)        vector<vector<double>>;
        %template(DoubleVector3)        vector<vector<vector<double>>>;
}

%include "DMF.hpp"
