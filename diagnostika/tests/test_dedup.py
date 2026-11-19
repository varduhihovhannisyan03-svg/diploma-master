import numpy as np
import pandas as pd

from src.data.dedup import choose_keepers, find_near_pairs, group_duplicates, hamming_to, sha256_file


def test_sha256_identical_bytes_match(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"same bytes")
    b.write_bytes(b"same bytes")
    assert sha256_file(a) == sha256_file(b)


def test_hamming_distance_is_popcount_of_xor():
    values = np.array([0, 0b1011, 2**63 + 1], dtype=np.uint64)
    assert hamming_to(values, 0).tolist() == [0, 3, 2]


def test_near_pairs_respect_threshold():
    hashes = np.array([0, 0b1111, 0b11111111_11111111], dtype=np.uint64)
    pairs = find_near_pairs(hashes, threshold=4)
    assert (0, 1, 4) in pairs
    assert not any(j == 2 for _, j, _ in pairs)  # distance 16 is not a near-duplicate


def test_exact_and_near_links_are_transitive():
    sha = ["x", "x", "y", "z"]
    phash = np.array([0, 2**60, 2**60 + 1, 2**64 - 1], dtype=np.uint64)  # 0~1 exact, 1~2 near (distance 1)
    groups = group_duplicates(sha, phash, threshold=4)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_group_spanning_train_and_test_keeps_test_copy_and_drops_train():
    df = pd.DataFrame({
        "image_id": ["TRAIN_A", "TEST_B", "TRAIN_C", "ONLY_TRAIN_D", "ONLY_TRAIN_E"],
        "official_split": ["train", "test", "train", "train", "train"],
        "group": [0, 0, 0, 1, 1],
    }, index=[10, 11, 12, 13, 14])
    out = choose_keepers(df)
    assert out.loc[11, "is_keeper"]                  # the test copy survives
    assert not out.loc[10, "is_keeper"]              # train copies are dropped
    assert not out.loc[12, "is_keeper"]
    assert out.loc[13, "is_keeper"] != out.loc[14, "is_keeper"]  # one keeper within a train-only group
    assert out.loc[[13, 14], "is_keeper"].sum() == 1
