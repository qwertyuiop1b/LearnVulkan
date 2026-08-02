#pragma once

#include "vk_pipeline.h"

#include <cstddef>

#include <glm/glm.hpp>

namespace triangle_example
{
struct Vertex
{
    glm::vec2 position;
    glm::vec3 color;

    static vk_engine::VertexInputDescription GetInputDescription()
    {
        vk_engine::VertexInputDescription description;

        vk::VertexInputBindingDescription binding{};
        binding.setBinding(0).setStride(sizeof(Vertex)).setInputRate(vk::VertexInputRate::eVertex);
        description.bindings.push_back(binding);

        vk::VertexInputAttributeDescription positionAttribute{};
        positionAttribute.setLocation(0)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32Sfloat)
            .setOffset(offsetof(Vertex, position));
        description.attributes.push_back(positionAttribute);

        vk::VertexInputAttributeDescription colorAttribute{};
        colorAttribute.setLocation(1)
            .setBinding(0)
            .setFormat(vk::Format::eR32G32B32Sfloat)
            .setOffset(offsetof(Vertex, color));
        description.attributes.push_back(colorAttribute);

        return description;
    }
};
} // namespace triangle_example
