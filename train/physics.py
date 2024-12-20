
def kmh_to_ms(speed):
    return int(speed // 3.6)

def ms_to_kmh(speed):
    return int(speed * 3.6)

def compute_security_distance(speed):
    return 1 if (speed < 1) else (speed // 10) ** 2
