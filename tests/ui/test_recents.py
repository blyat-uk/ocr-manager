"""`app/recents.py`: the welcome screen's recent files & folders -- the list
rules and the text each row shows. Qt-free; the QSettings glue and the rows
are tested in tests/ui/test_open_folder.py."""
from __future__ import annotations

import subprocess
import sys

import pytest

from app import recents
from app.recents import EPISODE, FOLDER, RECENTS_CAP, RecentEntry

DAY = 24 * 3600


def folder(path: str, opened: float = 0.0, count: int | None = 3) -> RecentEntry:
    return RecentEntry(path, FOLDER, opened, count)


def test_the_module_imports_no_qt():
    code = "import sys, app.recents; sys.exit(any(m.startswith('PyQt6') for m in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_recording_puts_the_entry_first_and_moves_one_already_listed():
    listed = recents.recorded([], folder("/a", 1))
    listed = recents.recorded(listed, folder("/b", 2))
    assert [entry.path for entry in listed] == ["/b", "/a"]
    listed = recents.recorded(listed, folder("/a", 3, count=5))
    assert listed == [folder("/a", 3, count=5), folder("/b", 2)]      # moved, and refreshed


def test_the_same_path_spelled_with_a_trailing_slash_is_one_entry():
    listed = recents.recorded([folder("/shows/A", 1)], folder("/shows/A/", 2))
    assert [entry.opened for entry in listed] == [2]


def test_the_list_keeps_the_newest_ten():
    listed = []
    for index in range(RECENTS_CAP + 3):
        listed = recents.recorded(listed, folder(f"/f{index}", index))
    assert len(listed) == RECENTS_CAP == 10
    assert listed[0].path == f"/f{RECENTS_CAP + 2}" and listed[-1].path == "/f3"


def test_without_drops_one_path():
    listed = [folder("/a"), folder("/b")]
    assert recents.without(listed, "/a") == [folder("/b")]
    assert recents.without(listed, "/nowhere") == listed


def test_dump_and_parse_round_trip():
    listed = [folder("/a", 10.5, 12), RecentEntry("/a/ep01.mkv", EPISODE, 9.0, None)]
    assert recents.parse(recents.dump(listed)) == listed


@pytest.mark.parametrize("text", ["", "not json", "{}", "[1, 2]", '[{"path": 3}]', "null"])
def test_parse_reads_anything_unreadable_as_an_empty_list(text):
    assert recents.parse(text) == []


def test_parse_skips_a_bad_entry_and_keeps_the_rest():
    text = '[{"path": "/a", "kind": "folder", "opened": 1, "count": 2}, {"path": "/b", "kind": "tape"}]'
    assert recents.parse(text) == [folder("/a", 1, 2)]


@pytest.mark.parametrize(("path", "title"), [
    ("/work/Martial Master", "Martial Master"),
    ("/work/Martial Master/Season 2", "Martial Master"),          # a season folder goes by its show
    ("/work/Show/S2 season", "Show"),
    ("/Season 1", "Season 1"),                                    # no ancestor names it: its own name
    ("C:\\Season 1", "Season 1"),                                 # a drive is never a show name
    ("/", "/"),
])
def test_a_folder_goes_by_the_nearest_name_that_says_which_show_it_is(path, title):
    assert recents.title(folder(path)) == title


def test_an_episode_goes_by_its_file_name_and_lives_in_its_folder():
    entry = RecentEntry("/work/Show/Show - 12 [4K].mkv", EPISODE, 0.0)
    assert recents.title(entry) == "Show - 12 [4K]"
    assert recents.location(entry) == "/work/Show"
    assert recents.location(folder("/work/Show")) == "/work/Show"


@pytest.mark.parametrize(("seconds", "text"), [
    (-5, "just now"), (0, "just now"), (59, "just now"),
    (60, "1 min ago"), (59 * 60, "59 min ago"),
    (3600, "1 h ago"), (23 * 3600, "23 h ago"),
    (DAY, "yesterday"), (2 * DAY, "2 days ago"), (6 * DAY, "6 days ago"),
    (7 * DAY, "1 week ago"), (34 * DAY, "4 weeks ago"),
    (35 * DAY, "1 month ago"), (359 * DAY, "11 months ago"),
    (360 * DAY, "1 year ago"), (800 * DAY, "2 years ago"),
])
def test_time_ago(seconds, text):
    assert recents.time_ago(seconds) == text


def test_the_detail_counts_a_folders_episodes_and_says_when():
    now = 1_000_000.0
    assert recents.detail(folder("/a", now - 7200, 12), now) == "12 episodes · 2 h ago"
    assert recents.detail(folder("/a", now, 1), now) == "1 episode · just now"
    assert recents.detail(folder("/a", now, 0), now) == "no episodes · just now"
    assert recents.detail(folder("/a", now, None), now) == "just now"
    assert recents.detail(RecentEntry("/a/x.mkv", EPISODE, now - DAY), now) == "yesterday"


def test_an_entry_exists_only_as_what_it_was(tmp_path):
    video = tmp_path / "ep01.mkv"
    video.write_bytes(b"")
    assert folder(str(tmp_path)).exists()
    assert RecentEntry(str(video), EPISODE, 0.0).exists()
    assert not RecentEntry(str(tmp_path), EPISODE, 0.0).exists()
    assert not folder(str(video)).exists()
    assert not folder(str(tmp_path / "gone")).exists()
