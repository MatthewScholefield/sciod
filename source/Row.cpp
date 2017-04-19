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
#include "Row.hpp"

Row::Row(int prevSize, int size) : nodes(size, Node(prevSize)) { }

size_t Row::numNodes() const
{
	return nodes.size();
}

size_t Row::numPrevNodes() const
{
	assert(nodes.size() > 0);
	return nodes[0].numLinks();
}

void Row::randomize()
{
	for (auto &i : nodes)
		i.randomize();
}

float Row::getBias(size_t id) const
{
	assert(id < nodes.size());
	return nodes[id].getBias();
}

void Row::updateBiases(const FloatVec &outputs, float learningRate)
{
	assert(outputs.size() == nodes.size());
	for (int i = 0; i < outputs.size(); ++i)
		nodes[i].updateBias(outputs[i], learningRate);
}

float &Row::getLinkRef(int src, int dest)
{
	assert(dest < nodes.size());
	return nodes[dest].getLinkRef(src);
}

float Row::getLink(int src, int dest) const
{
	assert(dest < nodes.size());
	return nodes[dest].getLink(src);
}

