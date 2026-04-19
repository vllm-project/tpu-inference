import socket

def get_host_ip():
    try:
        host_ip = socket.gethostbyname('host.docker.internal')
        return host_ip
    except socket.gaierror:
        print("Could not resolve host.docker.internal")
        return None