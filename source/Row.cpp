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

Row::Row(int size, int nextSize) : nodes(size, Node(nextSize)) { }

void Row::randomize()
{
	for (auto &i : nodes)
		i.randomize();
}

float Row::getLink(int src, int dest) const
{
	assert(src < nodes.size());
	return nodes[src].getLink(dest);
}

