#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <stdio.h>
int main()
{
    int serv_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    std::cout << "serv::" << serv_sock << std::endl;

    struct sockaddr_in serv_addr;
    memset( &serv_addr, 0, sizeof(serv_addr) );    

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = inet_addr("172.27.0.8");
    serv_addr.sin_port = htons(7000);
    int isBind =  bind(serv_sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    std::cout << "isBind::" << isBind << std::endl;
    
    int isListen =  listen(serv_sock,20);
    std::cout << "isListen::" << isListen << std::endl;

    struct sockaddr_in clnt_addr;
    memset( &clnt_addr, 0, sizeof(clnt_addr) );
    socklen_t clnt_addr_size = sizeof( clnt_addr ); 

    for(;;)
    {
        int clnt_sock = accept( serv_sock, ( struct sockaddr*)&clnt_addr,&clnt_addr_size );
        std::cout << "accept::" << clnt_sock << std::endl;

        {
            std::cout << "IP" << inet_ntoa( clnt_addr.sin_addr ) << std::endl;
	    std::cout << "PORT" << ntohs( clnt_addr.sin_port ) << std::endl;
        }

	sleep(10);
	std::cout << "PrintStart!" << std::endl;
        
        char str[] = "GOODBETTERBEST!";
        
        for( ;; )
	{
		sleep(5);
		write( clnt_sock, str, sizeof(str) );	
		std::cout << "Send New Message!" << std::endl;
 	}

	sleep(100);

        close(clnt_sock);
	std::cout << "OVER!" << std::endl;
    }
    close(serv_sock);
    return 0;
}
