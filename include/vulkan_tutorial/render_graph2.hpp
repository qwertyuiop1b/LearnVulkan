#pragma once

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace vulkan_tutorial {

struct Rg2Resource {
    uint32_t id = UINT32_MAX;
    std::string name;
};

struct Rg2Pass {
    std::string name;
    std::vector<uint32_t> reads;
    std::vector<uint32_t> writes;
    bool sideEffect = false;
    std::function<void(VkCommandBuffer)> record;
};

class RenderGraph2 {
  public:
    uint32_t addResource(const char* name) {
        resources_.push_back({static_cast<uint32_t>(resources_.size()), name});
        return resources_.back().id;
    }

    uint32_t addPass(Rg2Pass pass) {
        passes_.push_back(std::move(pass));
        return static_cast<uint32_t>(passes_.size() - 1);
    }

    void markOutput(uint32_t resource) { outputs_.push_back(resource); }

    bool compile() {
        const size_t count = passes_.size();
        std::vector<std::vector<uint32_t>> edges(count);
        std::vector<uint32_t> indegree(count, 0);
        std::unordered_map<uint32_t, uint32_t> producer;
        for (uint32_t pass = 0; pass < count; ++pass)
            for (uint32_t resource : passes_[pass].writes)
                producer[resource] = pass;
        for (uint32_t pass = 0; pass < count; ++pass) {
            auto connect = [&](uint32_t resource) {
                auto found = producer.find(resource);
                if (found == producer.end() || found->second == pass) return;
                edges[found->second].push_back(pass);
                ++indegree[pass];
            };
            for (uint32_t resource : passes_[pass].reads) connect(resource);
        }
        std::queue<uint32_t> ready;
        for (uint32_t i = 0; i < count; ++i)
            if (indegree[i] == 0) ready.push(i);
        order_.clear();
        while (!ready.empty()) {
            uint32_t pass = ready.front(); ready.pop();
            order_.push_back(pass);
            for (uint32_t next : edges[pass])
                if (--indegree[next] == 0) ready.push(next);
        }
        if (order_.size() != count) return false;

        std::vector<bool> live(count, false);
        std::vector<uint32_t> stack;
        for (uint32_t output : outputs_)
            for (uint32_t pass = 0; pass < count; ++pass)
                if (std::find(passes_[pass].writes.begin(), passes_[pass].writes.end(), output) !=
                    passes_[pass].writes.end()) stack.push_back(pass);
        for (uint32_t pass = 0; pass < count; ++pass)
            if (passes_[pass].sideEffect) stack.push_back(pass);
        while (!stack.empty()) {
            uint32_t pass = stack.back(); stack.pop_back();
            if (live[pass]) continue;
            live[pass] = true;
            for (uint32_t input : passes_[pass].reads)
                for (uint32_t candidate = 0; candidate < count; ++candidate)
                    if (std::find(passes_[candidate].writes.begin(), passes_[candidate].writes.end(), input) !=
                        passes_[candidate].writes.end()) stack.push_back(candidate);
        }
        liveOrder_.clear();
        for (uint32_t pass : order_)
            if (live[pass]) liveOrder_.push_back(pass);

        aliases_.clear();
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> lifetime;
        for (uint32_t order = 0; order < liveOrder_.size(); ++order) {
            const auto& pass = passes_[liveOrder_[order]];
            for (uint32_t resource : pass.reads) lifetime[resource].second = order;
            for (uint32_t resource : pass.writes) {
                auto& range = lifetime[resource];
                if (range.first == 0 && range.second == 0) range.first = order;
                range.second = order;
            }
        }
        for (const auto& resource : resources_) {
            auto range = lifetime.find(resource.id);
            if (range == lifetime.end()) continue;
            uint32_t alias = 0;
            while (alias < aliases_.size() && aliases_[alias].second >= range->second.second)
                ++alias;
            if (alias == aliases_.size())
                aliases_.push_back({range->second.first, range->second.second});
            else
                aliases_[alias].second = range->second.second;
            resourceAliases_[resource.id] = alias;
        }
        return true;
    }

    void record(VkCommandBuffer commandBuffer) const {
        for (uint32_t passIndex : liveOrder_) {
            if (passes_[passIndex].record) passes_[passIndex].record(commandBuffer);
        }
    }

    [[nodiscard]] const std::vector<uint32_t>& order() const { return order_; }
    [[nodiscard]] const std::vector<uint32_t>& liveOrder() const { return liveOrder_; }
    [[nodiscard]] size_t aliasCount() const { return aliases_.size(); }
    [[nodiscard]] uint32_t aliasOf(uint32_t resource) const {
        auto found = resourceAliases_.find(resource);
        return found == resourceAliases_.end() ? UINT32_MAX : found->second;
    }
    [[nodiscard]] const Rg2Pass& pass(uint32_t index) const { return passes_[index]; }

  private:
    std::vector<Rg2Resource> resources_;
    std::vector<Rg2Pass> passes_;
    std::vector<uint32_t> outputs_;
    std::vector<uint32_t> order_;
    std::vector<uint32_t> liveOrder_;
    std::vector<std::pair<uint32_t, uint32_t>> aliases_;
    std::unordered_map<uint32_t, uint32_t> resourceAliases_;
};

} // namespace vulkan_tutorial
