#pragma once

#include <vector>

namespace sciod
{
	using FloatVec = std::vector<float>;
	using FloatVec2D = std::vector<FloatVec>;

	FloatVec FloatVecSingle(float a);

	struct FloatVecIO
	{
		FloatVecIO(const FloatVec &in, const FloatVec &out);
		FloatVec in, out;
	};
}
