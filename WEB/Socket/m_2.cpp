#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
int main()
{
    int serv_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    std::cout << serv_sock << std::endl;

    struct sockaddr_in serv_addr;
    memset( &serv_addr, 0, sizeof(serv_addr) );
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("172.27.0.8");
    serv_addr.sin_port = htons(1567);
    int isBind =  bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    std::cout << "isBind::" << isBind << std::endl;

    int isListen =  listen(serv_sock,20);
    std::cout << "isListen::" << isListen << std::endl;

    struct sockaddr_in clnt_addr;
    socklen_t clnt_addr_size = sizeof( clnt_addr );
    int clnt_sock = accept( serv_sock, ( struct sockaddr*)&clnt_addr,&clnt_addr_size );
    std::cout << "accept::" << clnt_sock << std::endl;

    char str[] = "GOODBETTERBEST!";
    write( clnt_sock, str, sizeof(str) );

    close(clnt_sock);
    close(serv_sock);

    return 0;
}
