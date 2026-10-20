from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
RESULTS_PATH = REPORT_DIR / "retrieval_query_results.csv"
OUTPUT_PATH = REPORT_DIR / "retrieval_errors_top5.csv"

def main():
    results = pd.read_csv(RESULTS_PATH)
    required_columns = {'query', 'true_intent', 'first_relevant_rank', 'hit_rate_at_5'}
    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f'Retrieval results are missing columns : {sorted(missing)}'
        )
    errors = results[results['hit_rate_at_5'] == 0].copy()
    errors = errors.sort_values(by = ['true_intent', 'query']).reset_index(drop = True)
    errors.to_csv(OUTPUT_PATH, index = False)
    print(f'Validation queries : {len(results):,}')
    print(f'Queries with no correct intent hit in top 5 : {len(errors):,}')
    print(f'Failure rate : {len(errors) / len(results):.4%}')
    print(f'\nSaved failure cases to : {OUTPUT_PATH}')
    if errors.empty:
        print('No top 5 retrieval failures found.')
        return
    print('\nSample failures:')
    for row in errors.head(15).itertuples(index = False):
        print(f'\nQuery : {row.query}')
        print(f'True intent : {row.true_intent}')
        print(f'First relevant rank : {row.first_relevant_rank}')
        print('-' * 60)

if __name__ == '__main__':
    main()