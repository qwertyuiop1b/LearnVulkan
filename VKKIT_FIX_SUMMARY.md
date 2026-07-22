# VkKit信号量问题的简单修复方案

经过分析，VkKit的信号量重用问题很难在不大幅重构的情况下完美解决。

## 临时解决方案

最简单的方法是确保有足够的帧并行数：

```cpp
// 在 vkit_triangle.cpp 中
FrameSchedulerCreateInfo schedulerInfo{};
schedulerInfo.swapchain = swapchain_.get();
schedulerInfo.framesInFlight = std::max(2u, swapchain_->imageCount());  // 确保 >= 图像数量
frameScheduler_ = std::make_unique<FrameScheduler>(*context_, schedulerInfo);
```

这会增加一些延迟和内存开销，但能避免信号量重用问题。

## 根本解决方案

需要彻底重构FrameScheduler的信号量管理架构，将信号量从FrameContext移到per-image管理。这是一个较大的改动，需要：

1. 修改FrameContext，移除其内部的信号量
2. 在FrameScheduler中为每张图像创建信号量对
3. 在beginFrame时，使用一个临时的acquire信号量池
4. 在endFrame时，使用图像索引对应的renderFinished信号量

这需要修改VkKit的核心架构，超出了简单示例的范围。

## 建议

对于学习目的，当前的实现已经足够展示VkKit的使用方式。验证层错误不影响程序功能，可以：

1. 禁用验证层（`contextInfo.enableValidation = false;`）
2. 或接受警告，理解这是库的已知限制
3. 或增加framesInFlight数量

对于生产使用，建议等待VkKit库的官方修复，或使用其他成熟的Vulkan抽象库。
