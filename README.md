# MailLens — MVP интеллектуальной категоризации почтового трафика

**MailLens** — MVP-сервис интеллектуальной категоризации почтовых сообщений для корпоративного почтового потока.  
Решение поддерживает **zero-shot классификацию по произвольным пользовательским категориям**, обработку писем **.eml / .msg**, извлечение текста из **вложений**, порог **«Не определена»**, экспорт результатов и веб-интерфейс.

Проект реализован в рамках кейса ООО «АВ Софт» (информационная безопасность).

---

## Ключевые возможности

- Загрузка писем **.eml** (стабильно) и **.msg** (best-effort через `extract-msg`)
- Batch-обработка нескольких писем за один запрос
- Извлечение:
  - тема письма
  - тело письма
  - ссылки
  - вложения
- Вложения (MVP):
  - `pdf / docx / txt` → извлечение текста (с лимитами)
  - неизвестные форматы → безопасный skip без падений
- Мультиязычная **zero-shot классификация** (RU / EN)
- Классификация по **произвольным пользовательским категориям**
- Возврат **Top-N категорий** с оценками близости
- Категория **«Не определена»**, если `max_similarity < threshold`
- Базовые сигналы ИБ:
  - обнаружение prompt-инъекций
  - phishing risk score
- Экспорт результатов:
  - `results.csv`
- Web-интерфейс (Streamlit)
- API (FastAPI)
- Docker / docker-compose (API + UI)
- Логирование (`logs/app.log`)

---

mail_lens_repo/  
├── app/  
│   ├── __init__.py  
│   ├── api.py                 # FastAPI: /health, /classify/batch, /export/results.csv  
│   ├── classifier.py          # эмбеддинги + cosine + softmax + ранжирование категорий  
│   ├── embeddings.py          # загрузка sentence-transformers и embed_texts()  
│   ├── email_parser.py        # парсинг .eml/.msg + извлечение текста/метаданных  
│   ├── preprocessing.py       # очистка/нормализация текста (если используется)  
│   ├── security.py            # флаги prompt-injection/phishing risk (если используется)  
│   └── config.py              # settings: threshold/top_n/model/temperature и т.п.  
│  
├── ui/  
│   └── streamlit_app.py        # Streamlit UI: загрузка писем, таблица результатов, download CSV  
│  
├── data/  
│   └── samples/  
│       ├── test_finance.eml  
│       ├── test_phishing.eml  
│       └── test_with_attachment.eml  
│  
├── results/  
│   └── results.csv             # накопительный CSV-лог результатов (генерируется/обновляется)  
│  
├── logs/                       # логи приложения (опционально)  
│  
├── docker/  
│   ├── Dockerfile              # сборка образа (api/ui) под docker-compose  
│   └── docker-compose.yml      # запуск api (8000) + ui (8501)  
│  
├── tests/                      # тесты (опционально)  
│  
├── requirements.txt  
├── README.md  
└── ARCHITECTURE.md  


---

## Быстрый старт (Docker — рекомендуемый способ)

```bash
docker compose -f docker/docker-compose.yml up --build
```

После запуска:

- UI: http://localhost:8501  
- API healthcheck: http://localhost:8000/health  

---

## Запуск без Docker (dev-режим)

```bash
pip install -r requirements.txt

# API
uvicorn app.api:app --host 0.0.0.0 --port 8000

# UI (в другом терминале)
API_URL=http://localhost:8000 streamlit run ui/streamlit_app.py
```

---

## Как работает классификация

1. Письмо парсится и нормализуется  
2. Формируется единый текст:
   ```
   SUBJECT
   + BODY
   + ATTACHMENTS_TEXT
   + URLS
   ```
3. Строятся эмбеддинги письма и категорий  
4. Считается cosine similarity  
5. Формируется Top-N категорий  
6. Если `max_similarity < threshold` → категория **«Не определена»**

---

## Пояснение параметров

### Top-N категорий
Количество наиболее близких категорий, возвращаемых моделью. Используется для интерпретируемости результата.

### Threshold («Не определена»)
Минимальная уверенность классификации. Если значение ниже порога — письмо считается не относящимся ни к одной категории.

---

## Результаты и артефакты

- `results/results.csv`
- `results/results.jsonl`
- `logs/app.log`



