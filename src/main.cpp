#include "first_app.hpp"
#include <exception>
#include <iostream>
#include "demo.hpp"


int main() {

#ifdef BUILD_DEMO
    Demo demo{};
    try {
        demo.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
#else
    q_vulkan::FirstApp firstApp{};
    try {
        firstApp.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
#endif

}