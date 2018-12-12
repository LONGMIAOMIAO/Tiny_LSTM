#include <unistd.h>
#include <iostream>

int main()
{
    pid_t pid;

    pid = fork();

    switch( pid )
    {
    case 0:
        std::cout << "childID::" << pid << std::endl;
        break;
    default:
        sleep(3);
        std::cout << "father::" << pid << std::endl;
        break;
    }

    return 0;
}
