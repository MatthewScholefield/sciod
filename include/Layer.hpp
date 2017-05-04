#pragma once

#include <vector>
#include <cstdlib>

#include "Node.hpp"

namespace sciod
{

	class Layer
	{
	public:
		Layer(int prevSize, int size);
		size_t numNodes() const;
		size_t numPrevNodes() const;
		void randomize();
		float getBias(size_t id) const;
		void updateBiases(const FloatVec &outputs, float learningRate);
		float &getLinkRef(size_t src, size_t dest);
		float getLink(size_t src, size_t dest) const;

	private:
		std::vector<Node> nodes;
	};
}
