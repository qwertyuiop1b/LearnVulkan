#include "context.hpp"

int main() {
    custom::ContextInfo info{};
    custom::Context context(info);
    return context.init() ? 0 : 1;
}
