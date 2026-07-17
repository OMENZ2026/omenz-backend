import threading
import unittest

from memory import (
    ALLOWED_MEMORY_TYPES,
    MEMORY_VERSION,
    MemoryStore,
    normalize_confidence,
    normalize_memory_type,
)


# ==========================================================
# OMENZ MEMORY TESTS v0.1
# Validation suite for memory.py
# ==========================================================
#
# Purpose:
# - Validate controlled memory creation and retrieval.
# - Confirm revisions preserve history.
# - Confirm archival preserves records.
# - Confirm inactive-record visibility rules.
# - Validate type, confidence, limit, and status controls.
# - Confirm defensive-copy protection.
# - Confirm basic thread-safe creation behavior.
# - Confirm a new process-level store begins empty.
#
# Important:
# - Uses Python's built-in unittest framework.
# - Requires no new dependencies.
# - Does not modify main.py, router.py, or telemetry.py.
# - Does not call external provider APIs.
# ==========================================================


class MemoryStoreTestCase(unittest.TestCase):
    """
    Test the OMENZ MemoryStore using a fresh isolated store
    before every test.
    """

    def setUp(self) -> None:
        self.store = MemoryStore()

    def test_health_check_starts_empty(self) -> None:
        health = self.store.health_check()

        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["component"], "memory")
        self.assertEqual(health["version"], MEMORY_VERSION)
        self.assertEqual(health["storage"], "in_process")
        self.assertFalse(health["persistent"])
        self.assertEqual(health["total_records"], 0)
        self.assertEqual(health["active_records"], 0)
        self.assertEqual(health["archived_records"], 0)
        self.assertEqual(health["superseded_records"], 0)
        self.assertEqual(
            health["allowed_memory_types"],
            sorted(ALLOWED_MEMORY_TYPES),
        )

    def test_create_memory_record(self) -> None:
        record = self.store.create(
            memory_type="fact",
            content="The routing layer is operational.",
            source="integration_test",
            confidence=0.95,
            run_id="run-create-001",
            metadata={
                "department": "engineering",
            },
        )

        self.assertIsInstance(record["memory_id"], str)
        self.assertTrue(record["memory_id"])
        self.assertEqual(record["memory_version"], MEMORY_VERSION)
        self.assertEqual(record["memory_type"], "fact")
        self.assertEqual(
            record["content"],
            "The routing layer is operational.",
        )
        self.assertEqual(record["source"], "integration_test")
        self.assertEqual(record["confidence"], 0.95)
        self.assertEqual(record["status"], "active")
        self.assertEqual(record["run_id"], "run-create-001")
        self.assertEqual(
            record["metadata"]["department"],
            "engineering",
        )
        self.assertIsNone(record["archived_at"])
        self.assertIsNone(record["supersedes"])
        self.assertIsNone(record["superseded_by"])

    def test_create_normalizes_text_and_type(self) -> None:
        record = self.store.create(
            memory_type="  PREFERENCE  ",
            content="  Use complete file replacements.  ",
            source="  operator  ",
            confidence="0.8",
        )

        self.assertEqual(record["memory_type"], "preference")
        self.assertEqual(
            record["content"],
            "Use complete file replacements.",
        )
        self.assertEqual(record["source"], "operator")
        self.assertEqual(record["confidence"], 0.8)

    def test_get_active_memory_record(self) -> None:
        created = self.store.create(
            memory_type="task_state",
            content="Memory testing is active.",
            source="test_suite",
        )

        retrieved = self.store.get(created["memory_id"])

        self.assertIsNotNone(retrieved)
        self.assertEqual(
            retrieved["memory_id"],
            created["memory_id"],
        )
        self.assertEqual(
            retrieved["content"],
            "Memory testing is active.",
        )

    def test_get_unknown_memory_returns_none(self) -> None:
        retrieved = self.store.get("missing-memory-id")

        self.assertIsNone(retrieved)

    def test_create_rejects_invalid_memory_type(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create(
                memory_type="unknown_type",
                content="This should fail.",
                source="test_suite",
            )

    def test_create_rejects_empty_content(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create(
                memory_type="fact",
                content="   ",
                source="test_suite",
            )

    def test_create_rejects_empty_source(self) -> None:
        with self.assertRaises(ValueError):
            self.store.create(
                memory_type="fact",
                content="Source is required.",
                source="   ",
            )

    def test_confidence_accepts_valid_boundaries(self) -> None:
        low = self.store.create(
            memory_type="fact",
            content="Low-confidence record.",
            source="test_suite",
            confidence=0.0,
        )

        high = self.store.create(
            memory_type="fact",
            content="High-confidence record.",
            source="test_suite",
            confidence=1.0,
        )

        self.assertEqual(low["confidence"], 0.0)
        self.assertEqual(high["confidence"], 1.0)

    def test_confidence_rejects_invalid_values(self) -> None:
        invalid_values = [
            -0.01,
            1.01,
            "not-a-number",
            None,
        ]

        for value in invalid_values:
            with self.subTest(confidence=value):
                with self.assertRaises(ValueError):
                    normalize_confidence(value)

    def test_normalize_memory_type(self) -> None:
        self.assertEqual(
            normalize_memory_type("  SYSTEM_EVENT "),
            "system_event",
        )

        with self.assertRaises(ValueError):
            normalize_memory_type("unsupported")

    def test_list_active_records_by_default(self) -> None:
        first = self.store.create(
            memory_type="fact",
            content="First active fact.",
            source="test_suite",
        )

        second = self.store.create(
            memory_type="preference",
            content="Second active preference.",
            source="test_suite",
        )

        records = self.store.list_records()

        record_ids = {
            record["memory_id"]
            for record in records
        }

        self.assertEqual(len(records), 2)
        self.assertIn(first["memory_id"], record_ids)
        self.assertIn(second["memory_id"], record_ids)

    def test_list_records_filters_by_memory_type(self) -> None:
        self.store.create(
            memory_type="fact",
            content="Fact record.",
            source="test_suite",
        )

        preference = self.store.create(
            memory_type="preference",
            content="Preference record.",
            source="test_suite",
        )

        records = self.store.list_records(
            memory_type="preference",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0]["memory_id"],
            preference["memory_id"],
        )
        self.assertEqual(
            records[0]["memory_type"],
            "preference",
        )

    def test_list_records_rejects_invalid_status(self) -> None:
        with self.assertRaises(ValueError):
            self.store.list_records(
                status="deleted",
            )

    def test_list_records_rejects_invalid_limit(self) -> None:
        invalid_limits = [
            0,
            -1,
            "invalid",
        ]

        for value in invalid_limits:
            with self.subTest(limit=value):
                with self.assertRaises(ValueError):
                    self.store.list_records(limit=value)

    def test_revision_creates_new_record(self) -> None:
        original = self.store.create(
            memory_type="fact",
            content="Original statement.",
            source="operator",
            confidence=0.7,
            metadata={
                "topic": "routing",
            },
        )

        revised = self.store.revise(
            memory_id=original["memory_id"],
            content="Corrected statement.",
            source="operator_correction",
            confidence=0.95,
            run_id="run-revise-001",
            metadata={
                "reviewed": True,
            },
        )

        self.assertNotEqual(
            revised["memory_id"],
            original["memory_id"],
        )
        self.assertEqual(revised["status"], "active")
        self.assertEqual(
            revised["supersedes"],
            original["memory_id"],
        )
        self.assertEqual(
            revised["content"],
            "Corrected statement.",
        )
        self.assertEqual(
            revised["source"],
            "operator_correction",
        )
        self.assertEqual(revised["confidence"], 0.95)
        self.assertEqual(revised["run_id"], "run-revise-001")
        self.assertEqual(
            revised["metadata"]["topic"],
            "routing",
        )
        self.assertTrue(
            revised["metadata"]["reviewed"],
        )

    def test_revision_preserves_superseded_original(self) -> None:
        original = self.store.create(
            memory_type="fact",
            content="Original version.",
            source="test_suite",
        )

        revised = self.store.revise(
            memory_id=original["memory_id"],
            content="Revised version.",
            source="test_suite",
        )

        hidden_original = self.store.get(
            original["memory_id"],
        )

        visible_original = self.store.get(
            original["memory_id"],
            include_inactive=True,
        )

        self.assertIsNone(hidden_original)
        self.assertIsNotNone(visible_original)
        self.assertEqual(
            visible_original["status"],
            "superseded",
        )
        self.assertEqual(
            visible_original["superseded_by"],
            revised["memory_id"],
        )

    def test_revision_rejects_missing_record(self) -> None:
        with self.assertRaises(KeyError):
            self.store.revise(
                memory_id="missing-memory-id",
                content="New content.",
                source="test_suite",
            )

    def test_revision_rejects_inactive_record(self) -> None:
        original = self.store.create(
            memory_type="fact",
            content="Original version.",
            source="test_suite",
        )

        self.store.archive(
            memory_id=original["memory_id"],
            reason="No longer current.",
        )

        with self.assertRaises(ValueError):
            self.store.revise(
                memory_id=original["memory_id"],
                content="Attempted revision.",
                source="test_suite",
            )

    def test_archive_memory_record(self) -> None:
        created = self.store.create(
            memory_type="system_event",
            content="Temporary event.",
            source="test_suite",
        )

        archived = self.store.archive(
            memory_id=created["memory_id"],
            reason="Event lifecycle completed.",
            run_id="run-archive-001",
        )

        self.assertEqual(archived["status"], "archived")
        self.assertIsNotNone(archived["archived_at"])
        self.assertEqual(
            archived["metadata"]["archive_reason"],
            "Event lifecycle completed.",
        )

    def test_archived_record_is_hidden_by_default(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Record to archive.",
            source="test_suite",
        )

        self.store.archive(
            memory_id=created["memory_id"],
            reason="Testing visibility.",
        )

        hidden = self.store.get(created["memory_id"])

        visible = self.store.get(
            created["memory_id"],
            include_inactive=True,
        )

        self.assertIsNone(hidden)
        self.assertIsNotNone(visible)
        self.assertEqual(visible["status"], "archived")

    def test_archive_rejects_missing_record(self) -> None:
        with self.assertRaises(KeyError):
            self.store.archive(
                memory_id="missing-memory-id",
                reason="Testing.",
            )

    def test_archive_rejects_empty_reason(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Record to archive.",
            source="test_suite",
        )

        with self.assertRaises(ValueError):
            self.store.archive(
                memory_id=created["memory_id"],
                reason="   ",
            )

    def test_search_finds_active_records(self) -> None:
        matching = self.store.create(
            memory_type="preference",
            content="Use one file and one screenshot.",
            source="operator",
            confidence=0.9,
        )

        self.store.create(
            memory_type="fact",
            content="Unrelated information.",
            source="test_suite",
        )

        results = self.store.search("screenshot")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["memory_id"],
            matching["memory_id"],
        )

    def test_search_is_case_insensitive(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Cloud Run deployment completed.",
            source="test_suite",
        )

        results = self.store.search("CLOUD RUN")

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["memory_id"],
            created["memory_id"],
        )

    def test_search_filters_by_memory_type(self) -> None:
        self.store.create(
            memory_type="fact",
            content="Routing policy record.",
            source="test_suite",
        )

        preference = self.store.create(
            memory_type="preference",
            content="Routing policy preference.",
            source="test_suite",
        )

        results = self.store.search(
            query="routing",
            memory_type="preference",
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(
            results[0]["memory_id"],
            preference["memory_id"],
        )

    def test_search_excludes_archived_records(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Searchable archived content.",
            source="test_suite",
        )

        self.store.archive(
            memory_id=created["memory_id"],
            reason="Testing archived search behavior.",
        )

        results = self.store.search("searchable")

        self.assertEqual(results, [])

    def test_search_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            self.store.search("   ")

    def test_search_rejects_invalid_limit(self) -> None:
        invalid_limits = [
            0,
            -1,
            "invalid",
        ]

        for value in invalid_limits:
            with self.subTest(limit=value):
                with self.assertRaises(ValueError):
                    self.store.search(
                        query="test",
                        limit=value,
                    )

    def test_count_records_by_status(self) -> None:
        first = self.store.create(
            memory_type="fact",
            content="First record.",
            source="test_suite",
        )

        second = self.store.create(
            memory_type="fact",
            content="Second record.",
            source="test_suite",
        )

        self.store.archive(
            memory_id=first["memory_id"],
            reason="Testing count.",
        )

        self.store.revise(
            memory_id=second["memory_id"],
            content="Second record revised.",
            source="test_suite",
        )

        self.assertEqual(self.store.count(), 3)
        self.assertEqual(self.store.count("active"), 1)
        self.assertEqual(self.store.count("archived"), 1)
        self.assertEqual(self.store.count("superseded"), 1)

    def test_count_rejects_invalid_status(self) -> None:
        with self.assertRaises(ValueError):
            self.store.count("deleted")

    def test_returned_record_is_defensive_copy(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Protected record.",
            source="test_suite",
            metadata={
                "nested": {
                    "value": "original",
                },
            },
        )

        created["content"] = "Tampered content."
        created["metadata"]["nested"]["value"] = "tampered"

        stored = self.store.get(
            created["memory_id"],
        )

        self.assertEqual(
            stored["content"],
            "Protected record.",
        )
        self.assertEqual(
            stored["metadata"]["nested"]["value"],
            "original",
        )

    def test_retrieved_record_is_defensive_copy(self) -> None:
        created = self.store.create(
            memory_type="fact",
            content="Protected retrieval.",
            source="test_suite",
        )

        first_read = self.store.get(created["memory_id"])
        first_read["content"] = "Changed outside store."

        second_read = self.store.get(created["memory_id"])

        self.assertEqual(
            second_read["content"],
            "Protected retrieval.",
        )

    def test_metadata_input_is_defensively_copied(self) -> None:
        original_metadata = {
            "nested": {
                "value": "original",
            },
        }

        created = self.store.create(
            memory_type="fact",
            content="Metadata copy test.",
            source="test_suite",
            metadata=original_metadata,
        )

        original_metadata["nested"]["value"] = "changed"

        stored = self.store.get(created["memory_id"])

        self.assertEqual(
            stored["metadata"]["nested"]["value"],
            "original",
        )

    def test_concurrent_creation(self) -> None:
        thread_count = 20
        created_ids = []
        created_ids_lock = threading.Lock()

        def create_record(index: int) -> None:
            record = self.store.create(
                memory_type="system_event",
                content=f"Concurrent event {index}.",
                source="concurrency_test",
            )

            with created_ids_lock:
                created_ids.append(record["memory_id"])

        threads = [
            threading.Thread(
                target=create_record,
                args=(index,),
            )
            for index in range(thread_count)
        ]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        self.assertEqual(len(created_ids), thread_count)
        self.assertEqual(
            len(set(created_ids)),
            thread_count,
        )
        self.assertEqual(
            self.store.count("active"),
            thread_count,
        )

    def test_new_store_simulates_process_restart(self) -> None:
        self.store.create(
            memory_type="fact",
            content="Temporary in-process record.",
            source="test_suite",
        )

        replacement_store = MemoryStore()

        self.assertEqual(
            replacement_store.count(),
            0,
        )
        self.assertFalse(
            replacement_store.health_check()["persistent"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
