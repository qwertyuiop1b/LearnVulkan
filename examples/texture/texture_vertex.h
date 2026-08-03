#pragma once
#include "vk_pipeline.h"
#include <cstddef>
#include <glm/glm.hpp>
namespace texture_example
{
struct Vertex
{
    glm::vec2 position;
    glm::vec2 texCoord;
    static vk_engine::VertexInputDescription GetInputDescription()
    {
        vk_engine::VertexInputDescription description;
        description.bindings.push_back(
            vk::VertexInputBindingDescription{0, sizeof(Vertex), vk::VertexInputRate::eVertex});
        description.attributes.push_back(
            vk::VertexInputAttributeDescription{0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, position)});
        description.attributes.push_back(
            vk::VertexInputAttributeDescription{1, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord)});
        return description;
    }
};
} // namespace texture_example
