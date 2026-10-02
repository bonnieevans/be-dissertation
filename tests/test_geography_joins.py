"""Unit tests for the UPRN->MSOA11/MSOA21 and postcode->MSOA11/MSOA21 join
logic (src/05_attach_geography.py) using small synthetic fixtures.

MSOA11 is the PRIMARY analysis geography (see the module docstring in
05_attach_geography.py): transactions join postcode directly to a
currently-maintained postcode->MSOA11 lookup (NSPL "2011 Census"), and
new-builds join UPRN->postcode (via NSUL) then postcode->MSOA11 the same
way - a single-postcode-vintage path with no MSOA-boundary crosswalk step,
avoiding the merged-area ambiguity a MSOA21->MSOA11 crosswalk would create.
"""

import duckdb
import pytest


@pytest.fixture
def con():
    return duckdb.connect()


def test_uprn_left_join_preserves_row_count_and_flags_unmatched(con):
    con.execute("CREATE TABLE newbuilds AS SELECT * FROM (VALUES (1), (2), (3)) AS v(uprn)")
    con.execute("CREATE TABLE geog AS SELECT * FROM (VALUES (1, 'E02000001'), (2, 'E02000002')) AS v(uprn, msoa21cd)")
    result = con.sql(
        "SELECT n.uprn, g.msoa21cd FROM newbuilds n LEFT JOIN geog g USING (uprn) ORDER BY n.uprn"
    ).fetchall()
    assert len(result) == 3  # left join must not drop or duplicate rows
    assert result[2] == (3, None)  # uprn 3 has no geography match


def test_duplicate_uprn_in_lookup_does_not_fan_out_the_join(con):
    # NSUL is deduplicated with SELECT DISTINCT before the join in the real
    # pipeline; verify that without dedup a duplicate lookup row would fan
    # out matches, motivating that dedup step.
    con.execute("CREATE TABLE newbuilds AS SELECT * FROM (VALUES (1)) AS v(uprn)")
    con.execute(
        "CREATE TABLE geog_with_dupe AS SELECT * FROM (VALUES (1, 'E02000001'), (1, 'E02000001')) AS v(uprn, msoa21cd)"
    )
    fanned_out = con.sql("SELECT count(*) FROM newbuilds n JOIN geog_with_dupe g USING (uprn)").fetchone()[0]
    assert fanned_out == 2  # demonstrates the risk

    deduped = con.sql(
        "SELECT count(*) FROM newbuilds n JOIN (SELECT DISTINCT * FROM geog_with_dupe) g USING (uprn)"
    ).fetchone()[0]
    assert deduped == 1  # the pipeline's SELECT DISTINCT avoids the fan-out


def test_postcode_join_is_case_and_whitespace_insensitive(con):
    con.execute("CREATE TABLE trans AS SELECT * FROM (VALUES ('sl4 1qn'), ('SL4 1QN'), (' Sl4 1Qn ')) AS v(postcode)")
    con.execute("CREATE TABLE nspl AS SELECT * FROM (VALUES ('SL4 1QN', 'E02000001')) AS v(pcds, msoa21)")
    result = con.sql(
        """
        SELECT t.postcode, g.msoa21
        FROM trans t LEFT JOIN nspl g ON upper(trim(t.postcode)) = upper(trim(g.pcds))
        """
    ).fetchall()
    assert all(msoa == "E02000001" for _, msoa in result)


def test_match_rate_calculation(con):
    con.execute(
        "CREATE TABLE joined AS SELECT * FROM (VALUES ('E02000001'), ('E02000002'), (NULL)) AS v(msoa21cd)"
    )
    total, matched = con.sql(
        "SELECT count(*), sum(CASE WHEN msoa21cd IS NOT NULL THEN 1 ELSE 0 END) FROM joined"
    ).fetchone()
    match_rate = matched / total
    assert match_rate == pytest.approx(2 / 3)


def test_england_only_filter_drops_wales(con):
    # 05_attach_geography.py filters to MSOA21CD LIKE 'E02%' after geocoding.
    con.execute(
        "CREATE TABLE geo AS SELECT * FROM (VALUES ('E02000001'), ('W02000001'), ('E02000002')) AS v(msoa21cd)"
    )
    england_only = con.sql("SELECT msoa21cd FROM geo WHERE msoa21cd LIKE 'E02%'").fetchall()
    assert {r[0] for r in england_only} == {"E02000001", "E02000002"}


def test_uprn_to_msoa11_via_postcode_avoids_boundary_crosswalk(con):
    # New-builds: UPRN -> postcode (NSUL) -> MSOA11 (NSPL "2011 Census"),
    # never a MSOA21->MSOA11 boundary crosswalk. Two UPRNs sharing a
    # postcode both resolve to the same, unambiguous MSOA11.
    con.execute(
        "CREATE TABLE nsul AS SELECT * FROM (VALUES (1, 'AB1 0AA'), (2, 'AB1 0AA'), (3, 'CD2 0BB')) "
        "AS v(uprn, postcode)"
    )
    con.execute(
        "CREATE TABLE nspl_2011 AS SELECT * FROM (VALUES ('AB1 0AA', 'E02000001')) AS v(postcode, msoa11cd)"
    )
    result = con.sql(
        """
        SELECT n.uprn, p.msoa11cd
        FROM nsul n LEFT JOIN nspl_2011 p ON n.postcode = p.postcode
        ORDER BY n.uprn
        """
    ).fetchall()
    assert result == [(1, "E02000001"), (2, "E02000001"), (3, None)]


def test_msoa11_and_msoa21_are_independent_postcode_joins(con):
    # Both geographies are obtained by joining the SAME postcode against two
    # DIFFERENT lookups - not by crosswalking one MSOA vintage onto the
    # other - so a postcode with no MSOA21 entry can still resolve MSOA11
    # (and vice versa) without the two outcomes being coupled.
    con.execute("CREATE TABLE trans AS SELECT * FROM (VALUES ('AB1 0AA')) AS v(postcode)")
    con.execute("CREATE TABLE nspl_2011 AS SELECT * FROM (VALUES ('AB1 0AA', 'E02000001')) AS v(postcode, msoa11cd)")
    con.execute("CREATE TABLE nspl_2021 AS SELECT * FROM (VALUES ('AB1 0AA', NULL)) AS v(postcode, msoa21cd)")
    result = con.sql(
        """
        SELECT p11.msoa11cd, p21.msoa21cd
        FROM trans t
        LEFT JOIN nspl_2011 p11 ON t.postcode = p11.postcode
        LEFT JOIN nspl_2021 p21 ON t.postcode = p21.postcode
        """
    ).fetchone()
    assert result == ("E02000001", None)  # MSOA11 resolves even though MSOA21 doesn't
