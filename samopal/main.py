import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("output_file", default='YYMMDD-ч-ДЛИТ-ЧД.txt', nargs='?')
parser.add_argument("com_port", default="/dev/ttyACM0", nargs='?')
parser.add_argument("baud_rate", type=int, default=9600, nargs='?')
args = parser.parse_args()

# Параметры
max_points = 200 # размер графика
data = deque([0]*max_points, maxlen=max_points)
running = True # флаг завершения

# Чтение данных в отдельном потоке
def read_serial():
    global running
    try:
        with serial.Serial(args.com_port, args.baud_rate, timeout=1) as ser, open(args.output_file, 'a') as f:
            print(f"Чтение с порта {args.com_port} началось...")
            while running:
                line_raw = ser.readline().decode('utf-8', errors='ignore').strip()
                if line_raw:
                    try:
                        value = float(line_raw)
                        data.append(value)
                        f.write(f"{value}\n")
                    except ValueError:
                        pass # мусор откидываем
    except serial.SerialException as e:
        print(f"Ошибка при открытии порта: {e}")

# Запускаем поток чтения
thread = threading.Thread(target=read_serial)
thread.start()

# График
fig, ax = plt.subplots()
line, = ax.plot(data)
ax.set_ylim(400, 600) # под аналоговый вход

def update_plot(frame):
    line.set_ydata(data)
    line.set_xdata(range(len(data)))
    ax.relim()
    ax.autoscale_view()
    return line,

ani = animation.FuncAnimation(fig, update_plot, interval=50)
plt.title("Arduino Realtime Plotter")
plt.xlabel("№ записи")
plt.ylabel("Значение")
plt.grid(True)

try:
    plt.show()
finally:
    running = False
    thread.join()
    print("Программа завершена и порт закрыт.")
