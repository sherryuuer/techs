import argparse
import socket

MAX_SIZE_BYTES = 65535  # Mazimum size of a UDP datagram


def server(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    hostname = '127.0.0.1'
    s.bind((hostname, port))
    print('Listening at {}'.format(s.getsockname()))
    while True:
        data, clientAddress = s.recvfrom(MAX_SIZE_BYTES)
        message = data.decode('ascii')
        print(f'The client at {clientAddress} says {message}')
        replay = input('replay:')
        data = replay.encode('ascii')
        s.sendto(data, clientAddress)


def client(port):
    host = '127.0.0.1'
    port = 3000
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((host, port))
    client_name = input('What is your name? ')
    print('You can start conversition now!')
    while True:
        message = input(f'{client_name}: ')
        data = message.encode('ascii')
        s.send(data)
        data = s.recv(MAX_SIZE_BYTES)
        text = data.decode('ascii')
        print(f'The server replied with {text}')


if __name__ == '__main__':
    funcs = {'client': client, 'server': server}
    parser = argparse.ArgumentParser(description='UDP client and server')
    parser.add_argument('functions', choices=funcs, help='client or server')
    parser.add_argument('-p', metavar='PORT', type=int, default=3000,
                        help='UDP port (default 3000)')
    args = parser.parse_args()
    function = funcs[args.functions]
    function(args.p)
