#include "triangle_vertex.h"

#include <cassert>
#include <cstddef>

int main()
{
    const auto description = triangle_example::Vertex::GetInputDescription();

    assert(description.bindings.size() == 1);
    assert(description.attributes.size() == 2);

    const auto& binding = description.bindings.front();
    assert(binding.binding == 0);
    assert(binding.stride == sizeof(triangle_example::Vertex));
    assert(binding.inputRate == vk::VertexInputRate::eVertex);

    const auto& position = description.attributes[0];
    assert(position.location == 0);
    assert(position.binding == 0);
    assert(position.format == vk::Format::eR32G32Sfloat);
    assert(position.offset == offsetof(triangle_example::Vertex, position));

    const auto& color = description.attributes[1];
    assert(color.location == 1);
    assert(color.binding == 0);
    assert(color.format == vk::Format::eR32G32B32Sfloat);
    assert(color.offset == offsetof(triangle_example::Vertex, color));
}
