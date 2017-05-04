#include <cassert>
#include "sciod/Layer.hpp"

namespace sciod
{

Layer::Layer(int prevSize, int size) : nodes(size, Node(prevSize)) { }

size_t Layer::numNodes() const
{
	return nodes.size();
}

size_t Layer::numPrevNodes() const
{
	assert(nodes.size() > 0);
	return nodes[0].numLinks();
}

void Layer::randomize()
{
	for (auto &i : nodes)
		i.randomize();
}

float Layer::getBias(size_t id) const
{
	assert(id < nodes.size());
	return nodes[id].getBias();
}

void Layer::updateBiases(const FloatVec &outputs, float learningRate)
{
	assert(outputs.size() == nodes.size());
	for (size_t i = 0; i < outputs.size(); ++i)
		nodes[i].updateBias(outputs[i], learningRate);
}

float &Layer::getLinkRef(size_t src, size_t dest)
{
	assert(dest < nodes.size());
	return nodes[dest].getLinkRef(src);
}

float Layer::getLink(size_t src, size_t dest) const
{
	assert(dest < nodes.size());
	return nodes[dest].getLink(src);
}

}
