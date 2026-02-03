# Browser Agent

Автономный агент для выполнения веб‑задач в браузере с оркестрацией на LangGraph.

### Архитектура LangGraph

```
┌─────────────┐
│ Coordinator │ (Маршрутизатор между агентами)
└──────┬──────┘
       │
       ├───► Planner (Планирует следующий шаг)
       │
       ├───► Executor (Выполняет действия в браузере с помощью инструментов)
       │
       └───► Extractor (Анализирует содержимое страницы)
```


### Требования

- Python 3.11+
- Установленный браузер для Playwright

### Установка

```bash
pip install -r requirements.txt
playwright install chromium
```

### Конфигурация

Скопируйте `.env.example` в `.env` и заполните:

- `AGENTPLATFORM_KEY`
- `LLM_MODEL` (например `openai/gpt-4o`)
- `LLM_API_BASE_URL`
- `ENCRYPTION_KEY` (python -c "import base64; print(base64.b64encode(__import__('os').urandom(32)).decode())")

### Запуск

```bash
# интерактивный режим
python src/main.py
```

### Структура проекта

- `src/main.py` — консольный раннер
- `src/graph/*` — граф агента (координатор/планировщик/исполнитель/экстрактор)
- `src/tools/browser_tools.py` — инструменты Playwright
- `src/utils/session.py` — сохранение/загрузка зашифрованной сессии
