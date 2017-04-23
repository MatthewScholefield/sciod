/*
 * Copyright (C) 2017 Matthew D. Scholefield
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <cassert>
#include <cstdlib>
#include "Node.hpp"

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
