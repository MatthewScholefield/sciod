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
