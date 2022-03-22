from platform import uname

def in_wsl() -> bool:
    return 'microsoft-standard' in uname().release