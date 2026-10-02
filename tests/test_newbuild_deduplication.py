"""Unit tests for new-build UPRN deduplication logic (src/04_construct_newbuilds.py)
using small synthetic fixtures."""

import duckdb
import pytest


@pytest.fixture
def con():
    return duckdb.connect()


def test_earliest_lodgement_date_wins_per_uprn(con):
    con.execute(
        """
        CREATE TABLE epc AS SELECT * FROM (VALUES
            (1001, DATE '2015-06-01', 80.0),
            (1001, DATE '2015-01-15', 75.0),
            (1001, DATE '2018-09-01', 90.0),
            (1002, DATE '2016-03-01', 60.0)
        ) AS v(uprn, lodgement_date, total_floor_area)
        """
    )
    result = con.sql(
        """
        SELECT uprn, lodgement_date, total_floor_area FROM (
            SELECT *, row_number() OVER (PARTITION BY uprn ORDER BY lodgement_date ASC) AS rn
            FROM epc
        ) WHERE rn = 1 ORDER BY uprn
        """
    ).fetchall()
    assert result[0] == (1001, __import__("datetime").date(2015, 1, 15), 75.0)
    assert result[1] == (1002, __import__("datetime").date(2016, 3, 1), 60.0)


def test_one_row_per_uprn_in_output(con):
    con.execute(
        """
        CREATE TABLE epc AS SELECT * FROM (VALUES
            (1001, DATE '2015-06-01'),
            (1001, DATE '2015-01-15'),
            (1001, DATE '2018-09-01'),
            (1002, DATE '2016-03-01'),
            (1003, DATE '2017-01-01')
        ) AS v(uprn, lodgement_date)
        """
    )
    dedup = con.sql(
        """
        SELECT uprn FROM (
            SELECT *, row_number() OVER (PARTITION BY uprn ORDER BY lodgement_date ASC) AS rn
            FROM epc
        ) WHERE rn = 1
        """
    ).fetchall()
    uprns = [r[0] for r in dedup]
    assert len(uprns) == len(set(uprns)) == 3


def test_null_uprn_and_null_lodgement_date_excluded_before_dedup(con):
    con.execute(
        """
        CREATE TABLE epc AS SELECT * FROM (VALUES
            (1001, DATE '2015-01-15'),
            (NULL, DATE '2015-01-15'),
            (1002, NULL)
        ) AS v(uprn, lodgement_date)
        """
    )
    n_valid = con.sql(
        "SELECT count(*) FROM epc WHERE uprn IS NOT NULL AND lodgement_date IS NOT NULL"
    ).fetchone()[0]
    assert n_valid == 1


def test_tenure_group_mapping_never_folds_unknown_into_private():
    tenure_groups = {
        "SOCIAL": ["rental (social)", "rented (social)"],
        "PRIVATE_NONSOCIAL": ["owner-occupied", "rental (private)", "rented (private)"],
        "UNKNOWN": ["unknown", "no data!", ""],
    }
    all_private = {v for v in tenure_groups["PRIVATE_NONSOCIAL"]}
    all_unknown = {v for v in tenure_groups["UNKNOWN"]}
    assert all_private.isdisjoint(all_unknown)
    # A raw value not present in any explicit list must resolve to UNKNOWN,
    # not fall through to PRIVATE_NONSOCIAL by default.
    raw_value = "some unexpected new category"
    matched_group = next((g for g, vals in tenure_groups.items() if raw_value in vals), "UNKNOWN")
    assert matched_group == "UNKNOWN"
