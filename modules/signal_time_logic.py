# modules/signal_time_logic.py

def calculate_green_time(vehicle_count):
    min_time = 15
    max_time = 90
    time_per_vehicle = 2.5
    
    calculated_time = min_time + (vehicle_count * time_per_vehicle)
    
    if calculated_time < min_time: return int(min_time)
    elif calculated_time > max_time: return int(max_time)
    else: return int(calculated_time)

def get_signal_status_color(green_time):
    if green_time >= 50: return (0, 200, 0)
    elif green_time > 20: return (0, 165, 255)
    else: return (0, 0, 255)