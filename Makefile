install:
	uv sync

run:
	uv run python -m src --help

search_dataset-doc:
	uv run python -m src index --repo_path vllm-0.10.1
	uv run python -m src search_dataset data/datasets/public/AnsweredQuestions/dataset_docs_public.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/results_dataset_docs_public.json data/datasets/public/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

search_dataset-code:
	uv run python -m src index --repo_path vllm-0.10.1
	uv run python -m src search_dataset data/datasets/public/AnsweredQuestions/dataset_code_public.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/results_dataset_code_public.json data/datasets/public/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

prepare-moulinette:
	mkdir -p data/datasets/private/AnsweredQuestions data/datasets/private/UnansweredQuestions
	unzip -o datasets_private.zip
	cp datasets_public/public/AnsweredQuestions/dataset_code_public.json data/datasets/private/AnsweredQuestions/dataset_code_private.json
	cp datasets_public/public/AnsweredQuestions/dataset_docs_public.json data/datasets/private/AnsweredQuestions/dataset_docs_private.json
	cp datasets_public/public/UnansweredQuestions/dataset_code_public.json data/datasets/private/UnansweredQuestions/dataset_code_private.json
	cp datasets_public/public/UnansweredQuestions/dataset_docs_public.json data/datasets/private/UnansweredQuestions/dataset_docs_private.json
	rm -rf datasets_public/
	rm -rf datasets_private/
	


test-moulinette-recall:
	if [ ! -f data/processed/bm25_index/chunks.json ]; then uv run python -m src index --repo_path vllm-0.10.1; fi
	uv run python -m src search_dataset data/datasets/private/AnsweredQuestions/dataset_docs_private.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/results_dataset_docs_private.json data/datasets/private/AnsweredQuestions/dataset_docs_private.json --k 5 --max_context_length 2000
	uv run python -m src search_dataset data/datasets/private/AnsweredQuestions/dataset_code_private.json
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/results_dataset_code_private.json data/datasets/private/AnsweredQuestions/dataset_code_private.json --k 5 --max_context_length 2000

debug:
	uv run python -m pdb -m src --help

lint:
	uv run flake8 --exclude .venv,build,dist,vllm-0.10.1 src
	uv run mypy src --explicit-package-bases --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv run flake8 --exclude .venv,build,dist,vllm-0.10.1 src
	uv run mypy src --strict --explicit-package-bases --ignore-missing-imports

clean:
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf data/processed/bm25_index/* data/output/search_results/* data/output/answers/* datasets_public/

fclean: clean
	rm -rf .venv

.PHONY: install run debug lint lint-strict clean fclean