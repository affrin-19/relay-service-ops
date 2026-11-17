# Contributing

Contributions are welcome. Here is how to get set up and test your changes.

## Development Setup

1. Clone your fork and enter the directory:
   ```bash
   git clone https://github.com/<your-username>/relay-service-ops.git
   cd relay-service-ops
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

3. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

4. Run tests:
   ```bash
   python test_integration_review.py
   python test_stage_5_api_and_analytics.py
   python test_rag_pipeline.py
   ```

## Pull Request Guidelines

- Keep changes focused and small.
- Ensure all tests pass before submitting.
- Do not commit secrets, `.env` files, or database binaries.
- Follow standard Python conventions (PEP 8).
