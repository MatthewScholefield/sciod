#include "sciod/FloatVec.hpp"

namespace sciod
{
	FloatVec FloatVecSingle(float a)
	{
		return FloatVec(1, a);
	}

	FloatVecIO::FloatVecIO(const FloatVec& in, const FloatVec& out) :
	in(in), out(out) { }
}
