import json
from pathlib import Path

import pandas as pd
from pandas import json_normalize

DATA_DIR = Path(__file__).resolve().parent


def load_candidates_jsonl(path: Path) -> pd.DataFrame:
    """Load the main candidates dataset from a JSON Lines file."""
    return pd.read_json(path, lines=True)


def load_sample_candidates(path: Path) -> pd.DataFrame:
    """Load the sample candidates JSON array into a DataFrame."""
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_submission_csv(path: Path) -> pd.DataFrame:
    """Load the sample submission CSV file."""
    return pd.read_csv(path)


def flatten_candidate_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the top-level profile and redrob_signals objects for candidate-level analysis."""
    profile_df = json_normalize(df['profile']).add_prefix('profile.')
    redrob_df = json_normalize(df['redrob_signals']).add_prefix('redrob.')
    flat_df = pd.concat([df[['candidate_id']], profile_df, redrob_df], axis=1)
    return flat_df


def describe_nested_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns for nested arrays such as skills and career history."""
    result = df[['candidate_id']].copy()
    result['num_skills'] = df['skills'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    result['num_career_history_entries'] = df['career_history'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    result['num_education_entries'] = df['education'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    result['num_certifications'] = df['certifications'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    result['num_languages'] = df['languages'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return result


def main() -> None:
    candidates_path = DATA_DIR / 'candidates.jsonl'
    sample_candidates_path = DATA_DIR / 'sample_candidates.json'
    submission_path = DATA_DIR / 'sample_submission.csv'

    print('Loading datasets...')
    candidates_df = load_candidates_jsonl(candidates_path)
    sample_candidates_df = load_sample_candidates(sample_candidates_path)
    submission_df = load_submission_csv(submission_path)

    print('\n=== Dataset Summary ===')
    print(f'candidates.jsonl rows: {len(candidates_df):,}')
    print(f'sample_candidates.json rows: {len(sample_candidates_df):,}')
    print(f'sample_submission.csv rows: {len(submission_df):,}')

    print('\n=== candidates.jsonl columns ===')
    print(candidates_df.columns.tolist())

    print('\n=== sample_submission.csv columns ===')
    print(submission_df.columns.tolist())

    print('\n=== Flattened candidate sample ===')
    flat_df = flatten_candidate_fields(candidates_df)
    print(flat_df.head(3).to_string(index=False, max_colwidth=50))

    print('\n=== Nested counts sample ===')
    nested_counts_df = describe_nested_counts(candidates_df)
    print(nested_counts_df.head(5).to_string(index=False))

    print('\n=== sample_submission sample ===')
    print(submission_df.head(5).to_string(index=False))

    print('\n=== sample_candidates sample row ===')
    print(sample_candidates_df.head(1).to_dict(orient='records')[0])


if __name__ == '__main__':
    main()
