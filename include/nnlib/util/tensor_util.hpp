#ifndef UTIL_TENSOR_UTIL_HPP
#define UTIL_TENSOR_UTIL_HPP

#include "../core/tensor.hpp"

namespace nnlib
{

/// Output a tensor to a stream.
template <typename T>
std::ostream &operator<<(std::ostream &out, const Tensor<T> &t)
{
	out << std::left << std::setprecision(5) << std::fixed;
	
	if(t.dims() == 1)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			out << t(i) << "\n";
		}
	}
	else if(t.dims() == 2)
	{
		for(size_t i = 0; i < t.size(0); ++i)
		{
			for(size_t j = 0; j < t.size(1); ++j)
			{
				out << std::setw(10) << t(i, j);
			}
			out << "\n";
		}
	}
	
	out << "[ Tensor of dimension " << t.size(0);
	for(size_t i = 1; i < t.dims(); ++i)
	{
		out << " x " << t.size(i);
	}
	out << " ]";
	
	return out;
}

}

#endif
