
def to_seconds(time):
    seconds = int(time[:2]) * 3600 + int(time[2:4]) * 60 + int(time[4:])
    return seconds

def is_valid_interval(time1, time2, intv=180):
    time1 = to_seconds(time1)
    time2 = to_seconds(time2)
    
    difference = abs(time1 - time1)
    
    return difference >= intv
