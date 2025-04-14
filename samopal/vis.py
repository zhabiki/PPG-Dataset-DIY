import matplotlib.pyplot as plt

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input_file")
parser.add_argument("cutoff_start", type=int, default=5000, nargs='?')
parser.add_argument("cutoff_end", type=int, default=10000, nargs='?')
args = parser.parse_args()

# Считываем значения
data = []
with open(args.input_file, 'r') as f:
    for line in f:
        try:
            value = float(line.strip())
            data.append(value)
        except ValueError:
            pass # если строка не число - пропускаем

colors = ['magenta', 'red', 'blue']
for i in range(2):
    plt.figure(figsize=(10, 4))
    if (i == 1):
        plt.xlim(args.cutoff_start, args.cutoff_end)
        plt.ylim(300, 600)

    plt.plot(data, label=f'Окно {i+1}', color=colors[i-1])
    plt.title(f'График сигнала - окно {i+1}')
    plt.xlabel('№ записи')
    plt.ylabel('Значение')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

# Показываем все сразу
plt.show()
