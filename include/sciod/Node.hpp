#pragma once

#include <vector>
#include <cstdlib>

#include "sciod/FloatVec.hpp"

namespace sciod
{

	/*
	 * Represents a single point
	 * Contains connection weights to the previous nodes
	 */
	class Node
	{
	public:
		Node(int numConnections);
		size_t numLinks() const;
		void randomize();
		float getBias() const;
		void updateBias(float output, float learningRate);
		float &getLinkRef(size_t id);
		float getLink(size_t id) const;

	private:
		float bias;
		FloatVec weights;
	};
}
