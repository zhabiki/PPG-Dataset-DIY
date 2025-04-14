# PPG-Dataset-DIY

Код и инструкции, как самостоятельно и почти в домашних условиях состряпать свой собственный датасет ФПГ!


### Структура

- `samopal/` — Программы для снятия и визуализации данных для самодельного прибора на основе Arduino
- `swaid/` — Программа для снятия данных с предоставленных тестовых моделей приборов на основе SWAID


### Наша методика

1. Снимаем данные ФПГ с человека, как минимум 5 минут (в идеале, 10+ минут)
2. Даём человеку пройти серию небольших психологических тестов:
   - Шкала тревоги Бека (https://psytests.org/depr/bai.html) — 21 вопрос, 4 минуты
   - Шкала депрессии Бека (https://psytests.org/depr/bdi.html) — 21 вопрос, 4 минуты
   - Шкала HCL-33 д/БПД (https://psytests.org/diag/hcl32.html) — 33 вопроса, 6 минут
3. Пока человек проходит тесты, обрабатываем ФПГ ([см. основной репозиторий](https://github.com/zhabiki/PPG-Suicide-Inclinations))
4. Сопоставляем ВСР и нормализованные результаты тестов, заносим в базу. **Профит!**
