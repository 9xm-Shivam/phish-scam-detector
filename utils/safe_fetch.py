# utils/safe_fetch.py
import socket, ipaddress, requests
from urllib.parse import urlparse

def is_public_hostname(hostname):
    try:
        ip = socket.gethostbyname(hostname)
        ip_obj = ipaddress.ip_address(ip)
        return not (ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_reserved)
    except Exception:
        return False

def safe_get(url, timeout=5):
    parsed = urlparse(url)
    host = parsed.hostname
    if not host or not is_public_hostname(host):
        raise Exception("Refusing to fetch non-public or invalid host")
    r = requests.get(url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r.text
