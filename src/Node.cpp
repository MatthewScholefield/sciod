#include <cassert>
#include <cstdlib>
#include "sciod/Node.hpp"

namespace sciod
{

Node::Node(int numLinks) : weights(numLinks, 0.f) { }

static float randFloat(float min, float max)
{
	return min + ((max - min) * rand()) / RAND_MAX;
}

size_t Node::numLinks() const
{
	return weights.size();
}

void Node::randomize()
{
	for (float &i : weights)
		i = randFloat(-1.f, 1.f);
}

float Node::getBias() const
{
	return bias;
}

void Node::updateBias(float deriv, float learningRate)
{
	bias -= deriv * learningRate;
}

float &Node::getLinkRef(size_t id)
{
	assert(id < weights.size());
	return weights[id];
}

float Node::getLink(size_t id) const
{
	assert(id < weights.size());
	return weights[id];
}

}
