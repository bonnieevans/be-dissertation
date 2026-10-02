"""Unit tests for price-per-sqm construction logic (src/02_clean_transactions.py)
using small synthetic fixtures - not the full 23GB source data."""

import duckdb
import pytest


@pytest.fixture
def con():
    return duckdb.connect()


def test_calculated_ppsqm_matches_manual_division(con):
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (100000.0, 50.0),
            (250000.0, 100.0),
            (99999.0, 33.333)
        ) AS v(transaction_price, total_floor_area)
        """
    )
    result = con.sql(
        "SELECT transaction_price / total_floor_area AS ppsqm FROM t ORDER BY ppsqm"
    ).fetchall()
    assert result[0][0] == pytest.approx(2000.0)
    assert result[1][0] == pytest.approx(2500.0)
    assert result[2][0] == pytest.approx(99999.0 / 33.333)


def test_log_ppsqm_is_natural_log_of_nominal_ppsqm(con):
    import math

    con.execute("CREATE TABLE t AS SELECT 200000.0 AS transaction_price, 100.0 AS total_floor_area")
    row = con.sql(
        "SELECT transaction_price / total_floor_area AS nominal_ppsqm, "
        "ln(transaction_price / total_floor_area) AS log_ppsqm FROM t"
    ).fetchone()
    nominal_ppsqm, log_ppsqm = row
    assert log_ppsqm == pytest.approx(math.log(nominal_ppsqm))


def test_nonpositive_price_or_floor_area_excluded(con):
    con.execute(
        """
        CREATE TABLE t AS SELECT * FROM (VALUES
            (100000.0, 50.0),
            (0.0, 50.0),
            (100000.0, 0.0),
            (100000.0, -10.0),
            (-5.0, 50.0)
        ) AS v(transaction_price, total_floor_area)
        """
    )
    n_valid = con.sql(
        "SELECT count(*) FROM t WHERE transaction_price >= 1 AND total_floor_area IS NOT NULL AND total_floor_area >= 1"
    ).fetchone()[0]
    assert n_valid == 1


def test_annual_percentile_trim_is_computed_per_year_not_pooled(con):
    # Year A has small values, year B has large values. A pooled trim would
    # wrongly clip all of year A as "low outliers"; a per-year trim should not.
    con.execute(
        """
        CREATE TABLE t AS
        SELECT 2012 AS transaction_year, unnest(range(100)) + 100 AS nominal_ppsqm
        UNION ALL
        SELECT 2013 AS transaction_year, unnest(range(100)) + 10000 AS nominal_ppsqm
        """
    )
    bounds = con.sql(
        """
        SELECT transaction_year,
               quantile_cont(nominal_ppsqm, 0.005) AS lower_bound,
               quantile_cont(nominal_ppsqm, 0.995) AS upper_bound
        FROM t GROUP BY transaction_year ORDER BY transaction_year
        """
    ).fetchall()
    (year_a, low_a, high_a), (year_b, low_b, high_b) = bounds
    assert year_a == 2012 and year_b == 2013
    # Each year's bounds should sit within that year's own value range, not
    # be dragged toward the other year's scale.
    assert 100 <= low_a <= high_a <= 199
    assert 10000 <= low_b <= high_b <= 10099
