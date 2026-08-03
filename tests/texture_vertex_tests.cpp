#include "texture_vertex.h"
#include <cassert>
#include <cstddef>
int main()
{
    const vk_engine::VertexInputDescription description = texture_example::Vertex::GetInputDescription();
    assert(description.bindings.size() == 1);
    assert(description.bindings[0].binding == 0);
    assert(description.bindings[0].stride == sizeof(texture_example::Vertex));
    assert(description.attributes.size() == 2);
    assert(description.attributes[0].location == 0);
    assert(description.attributes[0].format == vk::Format::eR32G32Sfloat);
    assert(description.attributes[0].offset == offsetof(texture_example::Vertex, position));
    assert(description.attributes[1].location == 1);
    assert(description.attributes[1].format == vk::Format::eR32G32Sfloat);
    assert(description.attributes[1].offset == offsetof(texture_example::Vertex, texCoord));
    return 0;
}
