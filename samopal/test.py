import serial
import serial.tools.list_ports

ser = None

# Показываем список доступных портов
print("Доступные порты: ")
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"{port.device} - {port.description}")
