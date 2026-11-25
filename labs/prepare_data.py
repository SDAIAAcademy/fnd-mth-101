# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2.1"]
# ///
"""Write the synthetic data of the online learning platform into labs/data/.

    cd labs
    uv run prepare_data.py

Fixed seed (101): the same bytes on every run. Every number is written with at most six
decimals. The platform, its learners and its courses are fictional: learners, courses, help
articles and help-centre pages are known by code only (L0001, C01, A001, P01).

Planted problems, each found by a named lab (see labs/solutions/answer_key.md):
- learner_features.csv: total_actions = videos + quizzes + posts exactly (rank deficiency, lab 3.1)
- learner_features.csv: video_minutes is almost 7.5 x videos (near dependence, lab 4.1)
- study_hours.csv: video_minutes and video_count nearly collinear (conditioning, lab 5.1)
- help_links.csv: a dangling page, a three-page link farm and a five-page island (lab 6.1)
- ratings.csv: true rank 3 plus noise, with a held-out set (lab 8.1)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

SEED = 101
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
REF = DATA / "reference"

ARTICLES = {
    "account": [
        "reset my password", "change the email on my account", "delete my account for good",
        "sign in on a new phone", "my account is locked after many tries", "change my display name",
        "turn on two step sign in", "i forgot which email i used", "merge two accounts into one",
        "update my date of birth",
    ],
    "certificates": [
        "download my course certificate", "my certificate shows the wrong name", "when will my certificate arrive",
        "share a certificate with an employer", "print a certificate on paper", "certificate missing after i passed",
        "check that a certificate is genuine", "certificate for a course i finished last year",
        "the date on my certificate is wrong", "get a certificate in arabic",
    ],
    "video": [
        "video will not play", "video keeps stopping to load", "turn on subtitles for a video",
        "change the speed of a video", "download a video to watch later", "the sound is out of step with the picture",
        "video is too dark to see", "watch videos on a slow connection", "video quality is low", "the video player shows an error",
    ],
    "quizzes": [
        "retake a quiz i failed", "my quiz answers were not saved", "how many tries do i get on a quiz",
        "the quiz timer ran out early", "see the correct answers after a quiz", "a quiz question looks wrong",
        "quiz score is missing from my grades", "start a quiz again from the beginning", "the quiz will not submit",
        "what score do i need to pass",
    ],
    "progress": [
        "my progress was reset", "mark a lesson as complete", "progress bar is stuck", "see how much of the course is left",
        "progress not saved on my phone", "move back to an earlier lesson", "my finished lessons show as new",
        "set a weekly study goal", "see my study time this week", "pick up where i stopped",
    ],
    "enrolment": [
        "join a course", "leave a course i joined", "the course is full", "join a course that has started",
        "find courses for beginners", "a course i joined has disappeared", "change to another group of the same course",
        "when does the next course start", "join a course with my team", "courses that need no experience",
    ],
}

QUERIES = [
    ("Q01", "i cannot remember my password", "A001"),
    ("Q02", "the film stops all the time", "A022"),
    ("Q03", "how do i get my certificate", "A011"),
    ("Q04", "i want to take the test again", "A031"),
    ("Q05", "my lessons are marked as not done", "A047"),
    ("Q06", "sign up for a class", "A051"),
    ("Q07", "my name is spelled wrong on the certificate", "A012"),
    ("Q08", "subtitles for the video", "A023"),
    ("Q09", "remove my account", "A003"),
    ("Q10", "the test clock ended too soon", "A034"),
]

OCCUPATIONS = ["nurse", "engineer", "teacher", "programmer", "doctor", "receptionist", "pilot",
               "secretary", "scientist", "librarian", "mechanic", "designer", "housekeeper", "carpenter"]
GENDER_PAIRS = [["she", "he"], ["woman", "man"], ["her", "his"], ["female", "male"], ["mother", "father"]]


def r6(x: float) -> float:
    return float(f"{x:.6f}")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def articles() -> None:
    rows, k = [], 1
    for topic, titles in ARTICLES.items():
        for t in titles:
            rows.append([f"A{k:03d}", topic, t])
            k += 1
    write_csv(DATA / "help_articles.csv", ["article", "topic", "title"], rows)
    write_csv(DATA / "queries.csv", ["query", "text", "best_article"], [list(q) for q in QUERIES])
    write_json(DATA / "bias_words.json", {"occupations": OCCUPATIONS, "gender_pairs": GENDER_PAIRS})


def learner_features(rng: np.random.Generator) -> None:
    n = 400
    videos = rng.poisson(18, n) + 1
    minutes = np.round(7.5 * videos + rng.normal(0, 2.0, n), 1)  # nearly 7.5 x videos
    quizzes = rng.poisson(6, n)
    posts = rng.poisson(2, n)
    total = videos + quizzes + posts  # exactly the sum: a redundant column
    weekly = np.round(0.9 * minutes / 7.5 + 1.6 * quizzes + 0.8 * posts + rng.normal(0, 3, n) + 5, 1)
    rows = [[f"L{i + 1:04d}", int(videos[i]), r6(minutes[i]), int(quizzes[i]), int(posts[i]), int(total[i]), r6(weekly[i])]
            for i in range(n)]
    write_csv(DATA / "learner_features.csv",
              ["learner", "videos", "video_minutes", "quizzes", "posts", "total_actions", "weekly_minutes"], rows)


def study_hours(rng: np.random.Generator) -> None:
    n = 120
    count = rng.integers(6, 40, n).astype(float)
    minutes = 7.5 * count + rng.normal(0, 0.6, n)  # nearly collinear with count
    quizzes = rng.integers(2, 12, n).astype(float)
    hours = 0.05 * minutes + 0.4 * count + 0.9 * quizzes + 2.0 + rng.normal(0, 0.5, n)
    rows = [[f"C{i + 1:03d}", r6(count[i]), r6(minutes[i]), r6(quizzes[i]), r6(hours[i])] for i in range(n)]
    write_csv(DATA / "study_hours.csv", ["course", "video_count", "video_minutes", "quizzes", "hours_to_complete"], rows)


def help_links(rng: np.random.Generator) -> None:
    # P01 to P30: the main help centre, with P01 the home page and P02 to P07 topic hubs
    edges: set[tuple[str, str]] = set()
    main = [f"P{i:02d}" for i in range(1, 31)]
    hubs = main[1:7]
    for h in hubs:
        edges.add(("P01", h))
        edges.add((h, "P01"))
    for i, p in enumerate(main[7:], start=7):
        hub = hubs[i % 6]
        edges.add((hub, p))
        edges.add((p, hub))
        for q in rng.choice(main[7:], size=2, replace=False):
            if q != p:
                edges.add((p, str(q)))
    # P31: a page promoted by a link farm P32 to P34, which link to each other and to P31
    farm = ["P32", "P33", "P34"]
    for a in farm:
        edges.add((a, "P31"))
        for b in farm:
            if a != b:
                edges.add((a, b))
    edges.add(("P31", "P32"))
    edges.add(("P12", "P31"))  # one ordinary page links into the farm's target
    # P35: a dangling page, linked to but linking nowhere (an old PDF guide)
    edges.add(("P05", "P35"))
    edges.add(("P20", "P35"))
    # P36 to P40: an island of archived pages linked only to each other
    island = ["P36", "P37", "P38", "P39", "P40"]
    for i, a in enumerate(island):
        edges.add((a, island[(i + 1) % len(island)]))
    rows = sorted([list(e) for e in edges])
    write_csv(DATA / "help_links.csv", ["from_page", "to_page"], rows)


def ratings(rng: np.random.Generator) -> None:
    n_l, n_c, k = 400, 30, 3
    u = rng.normal(0, 1, (n_l, k))
    v = rng.normal(0, 1, (n_c, k))
    truth = 3.4 + 0.55 * (u @ v.T) / np.sqrt(k) + rng.normal(0, 0.25, (n_l, n_c))
    truth = np.clip(np.round(truth * 2) / 2, 1, 5)  # half-star ratings from 1 to 5
    observed = rng.random((n_l, n_c)) < 0.3
    held = observed & (rng.random((n_l, n_c)) < 0.2)
    rows = []
    for i in range(n_l):
        for j in range(n_c):
            if observed[i, j]:
                rows.append([f"L{i + 1:04d}", f"C{j + 1:03d}", r6(truth[i, j]), "test" if held[i, j] else "train"])
    write_csv(DATA / "ratings.csv", ["learner", "course", "rating", "split"], rows)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    REF.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    articles()
    learner_features(rng)
    study_hours(rng)
    help_links(rng)
    ratings(rng)
    print("wrote", sorted(p.name for p in DATA.glob("*.*")))


if __name__ == "__main__":
    main()
