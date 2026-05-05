#!/bin/bash
EXTENSIONS="cpp|cc|cxx|h|hpp|hxx"
find ./src ./include -type f -regextype posix-extended -iregex ".*\.(${EXTENSIONS})" -print0 | xargs -0 clang-format -i -style=file