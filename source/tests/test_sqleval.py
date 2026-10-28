"""Tests for the execution-accuracy scorer, run against real SQLite databases."""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sqleval import run_query, score_one, has_order_by, compare_rows  # noqa: E402
from prompt import extract_sql  # noqa: E402


def make_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE singer (
            singer_id INTEGER PRIMARY KEY, name TEXT,
            country TEXT, age INTEGER, net_worth REAL
        );
        CREATE TABLE concert (
            concert_id INTEGER PRIMARY KEY, singer_id INTEGER,
            city TEXT, year INTEGER
        );
        INSERT INTO singer VALUES
            (1,'Ana','France',34,2.5),(2,'Bo','France',52,1.0),
            (3,'Cy','Japan',29,3.0),(4,'Di','Japan',45,0.5),
            (5,'Eve','Brazil',29,7.25);
        INSERT INTO concert VALUES
            (1,1,'Paris',2021),(2,1,'Lyon',2022),(3,3,'Tokyo',2022),
            (4,5,'Rio',2023),(5,5,'Rio',2024);
    """)
    conn.commit()
    conn.close()


class TestScorer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "music.sqlite"
        make_db(cls.db)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def score(self, gold, pred):
        return score_one(str(self.db), gold, pred)

    # --- the core claim: semantically equal SQL scores correct -------------
    def test_alias_and_qualification_differences_still_correct(self):
        d = self.score(
            "SELECT name FROM singer WHERE age > 30",
            "SELECT s.name FROM singer AS s WHERE s.age >= 31",
        )
        self.assertTrue(d.correct, d.reason)

    def test_row_order_ignored_when_gold_has_no_order_by(self):
        d = self.score(
            "SELECT name FROM singer WHERE country = 'Japan'",
            "SELECT name FROM singer WHERE country = 'Japan' ORDER BY name DESC",
        )
        self.assertTrue(d.correct, d.reason)

    def test_row_order_enforced_when_gold_has_order_by(self):
        d = self.score(
            "SELECT name FROM singer ORDER BY age ASC",
            "SELECT name FROM singer ORDER BY age DESC",
        )
        self.assertFalse(d.correct, d.reason)

    def test_int_vs_real_not_penalised(self):
        d = self.score(
            "SELECT COUNT(*) FROM singer",
            "SELECT SUM(1.0) FROM singer",
        )
        self.assertTrue(d.correct, d.reason)

    def test_column_order_difference_tolerated(self):
        d = self.score(
            "SELECT name, country FROM singer WHERE age = 29",
            "SELECT country, name FROM singer WHERE age = 29",
        )
        self.assertTrue(d.correct, d.reason)

    # --- the other core claim: wrong SQL scores wrong ----------------------
    def test_wrong_filter_is_incorrect(self):
        d = self.score(
            "SELECT name FROM singer WHERE country = 'France'",
            "SELECT name FROM singer WHERE country = 'Japan'",
        )
        self.assertFalse(d.correct, d.reason)

    def test_duplicates_matter(self):
        d = self.score(
            "SELECT city FROM concert",
            "SELECT DISTINCT city FROM concert",
        )
        self.assertFalse(d.correct, d.reason)

    def test_missing_join_is_incorrect(self):
        d = self.score(
            "SELECT s.name FROM singer s JOIN concert c ON s.singer_id=c.singer_id "
            "WHERE c.city='Rio'",
            "SELECT name FROM singer",
        )
        self.assertFalse(d.correct, d.reason)

    def test_syntax_error_reports_not_correct(self):
        d = self.score("SELECT name FROM singer", "SELECT nmae FROM singer")
        self.assertFalse(d.correct)
        self.assertIsNotNone(d.pred_error)

    def test_empty_prediction(self):
        d = self.score("SELECT name FROM singer", "")
        self.assertFalse(d.correct)

    def test_broken_gold_is_flagged_not_blamed_on_model(self):
        d = self.score("SELECT * FROM no_such_table", "SELECT name FROM singer")
        self.assertFalse(d.correct)
        self.assertIsNotNone(d.gold_error)

    # --- safety -----------------------------------------------------------
    def test_destructive_prediction_is_refused(self):
        r = run_query(str(self.db), "DROP TABLE singer")
        self.assertFalse(r.ok)
        conn = sqlite3.connect(self.db)
        n = conn.execute("SELECT COUNT(*) FROM singer").fetchone()[0]
        conn.close()
        self.assertEqual(n, 5, "database must survive a destructive prediction")

    def test_readonly_connection_blocks_writes(self):
        r = run_query(str(self.db), "DELETE FROM singer", allow_writes=True)
        self.assertFalse(r.ok)

    # --- helpers ----------------------------------------------------------
    def test_order_by_inside_string_literal_not_detected(self):
        self.assertFalse(has_order_by("SELECT * FROM t WHERE x = 'order by'"))
        self.assertTrue(has_order_by("SELECT * FROM t ORDER BY x"))

    def test_compare_handles_mixed_types_without_crashing(self):
        ok, _ = compare_rows([(1, "a"), ("b", 2)], [("b", 2), (1, "a")], False)
        self.assertTrue(ok)


class TestExtractSQL(unittest.TestCase):
    def test_strips_markdown_fence(self):
        self.assertEqual(
            extract_sql("```sql\nSELECT a FROM t;\n```"), "SELECT a FROM t")

    def test_strips_prose_preamble(self):
        self.assertEqual(
            extract_sql("Sure! Here is the query:\nSELECT a FROM t;"),
            "SELECT a FROM t")

    def test_keeps_only_first_statement(self):
        self.assertEqual(
            extract_sql("SELECT a FROM t; SELECT b FROM u;"), "SELECT a FROM t")

    def test_handles_cte(self):
        self.assertTrue(extract_sql("WITH x AS (SELECT 1) SELECT * FROM x")
                        .startswith("WITH"))

    def test_plain_sql_unchanged(self):
        self.assertEqual(extract_sql("SELECT 1"), "SELECT 1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
