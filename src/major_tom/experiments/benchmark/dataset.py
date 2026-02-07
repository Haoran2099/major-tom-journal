"""Benchmark dataset management for ACM MM experiments."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import logging
import random
import time

logger = logging.getLogger(__name__)


@dataclass
class ActivityEvent:
    """Single activity event in the benchmark."""

    timestamp: float
    app: str
    title: str
    duration_seconds: float
    kpm: float  # keystrokes per minute
    cpm: float  # clicks per minute

    # Ground truth annotations
    activity_type: str
    importance_score: int  # 1-5

    # Optional
    reference_summary: Optional[str] = None
    screenshot_path: Optional[str] = None


@dataclass
class ActivitySession:
    """A session of continuous activity events."""

    session_id: str
    events: List[ActivityEvent] = field(default_factory=list)

    # Session-level annotations
    session_summary: str = ""
    qa_pairs: List[Tuple[str, str]] = field(default_factory=list)
    importance_ranking: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "session_id": self.session_id,
            "events": [asdict(e) for e in self.events],
            "session_summary": self.session_summary,
            "qa_pairs": self.qa_pairs,
            "importance_ranking": self.importance_ranking,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ActivitySession":
        """Create from dictionary."""
        events = [ActivityEvent(**e) for e in data.get("events", [])]
        return cls(
            session_id=data["session_id"],
            events=events,
            session_summary=data.get("session_summary", ""),
            qa_pairs=data.get("qa_pairs", []),
            importance_ranking=data.get("importance_ranking", []),
        )


# Activity profiles for realistic synthetic generation
ACTIVITY_PROFILES = {
    "coding": {
        "apps": [
            ("VS Code", ["main.py - project", "utils.py - project", "test_main.py - project", "README.md - project"]),
            ("Terminal", ["zsh", "python main.py", "git status", "npm run dev"]),
            ("Xcode", ["ViewController.swift", "AppDelegate.swift", "Main.storyboard"]),
        ],
        "kpm_range": (80, 200),
        "cpm_range": (5, 30),
        "duration_range": (120, 600),
        "importance_range": (3, 5),
        "weight": 0.25,
    },
    "web_research": {
        "apps": [
            ("Safari", ["arxiv.org - Paper Title", "Google Scholar", "Stack Overflow - Question", "GitHub - Repository"]),
            ("Chrome", ["Wikipedia - Topic", "MDN Web Docs", "Research Paper PDF"]),
        ],
        "kpm_range": (20, 80),
        "cpm_range": (30, 80),
        "duration_range": (60, 300),
        "importance_range": (2, 4),
        "weight": 0.20,
    },
    "document_writing": {
        "apps": [
            ("Pages", ["Research Report.pages", "Meeting Notes.pages"]),
            ("Notes", ["Project Ideas", "Daily Notes", "Meeting Notes"]),
            ("Obsidian", ["Knowledge Base", "Literature Notes", "Project Planning"]),
            ("Word", ["Report.docx", "Proposal.docx"]),
        ],
        "kpm_range": (60, 150),
        "cpm_range": (10, 40),
        "duration_range": (180, 900),
        "importance_range": (3, 5),
        "weight": 0.15,
    },
    "email_communication": {
        "apps": [
            ("Mail", ["Inbox", "Sent", "RE: Project Update", "FW: Meeting"]),
            ("Outlook", ["Inbox - Outlook", "Calendar - Outlook"]),
        ],
        "kpm_range": (40, 120),
        "cpm_range": (20, 50),
        "duration_range": (30, 180),
        "importance_range": (2, 4),
        "weight": 0.10,
    },
    "video_watching": {
        "apps": [
            ("Safari", ["YouTube - Video Title", "Bilibili - Video", "Netflix"]),
            ("IINA", ["movie.mp4", "lecture.mp4"]),
        ],
        "kpm_range": (0, 10),
        "cpm_range": (5, 20),
        "duration_range": (300, 1800),
        "importance_range": (1, 3),
        "weight": 0.08,
    },
    "social_media": {
        "apps": [
            ("Safari", ["Twitter", "Reddit - r/programming", "Hacker News"]),
            ("WeChat", ["WeChat"]),
            ("Telegram", ["Telegram"]),
        ],
        "kpm_range": (20, 80),
        "cpm_range": (40, 100),
        "duration_range": (30, 300),
        "importance_range": (1, 2),
        "weight": 0.07,
    },
    "file_management": {
        "apps": [
            ("Finder", ["Documents", "Downloads", "Desktop"]),
        ],
        "kpm_range": (0, 20),
        "cpm_range": (30, 80),
        "duration_range": (15, 120),
        "importance_range": (1, 2),
        "weight": 0.05,
    },
    "meeting": {
        "apps": [
            ("Zoom", ["Zoom Meeting", "Team Sync"]),
            ("FaceTime", ["FaceTime Call"]),
            ("Slack", ["#general - Slack", "Direct Message"]),
        ],
        "kpm_range": (10, 60),
        "cpm_range": (10, 30),
        "duration_range": (900, 3600),
        "importance_range": (3, 5),
        "weight": 0.04,
    },
    "gaming": {
        "apps": [
            ("Steam", ["Game Title"]),
            ("App Store", ["Game"]),
        ],
        "kpm_range": (50, 200),
        "cpm_range": (100, 300),
        "duration_range": (600, 3600),
        "importance_range": (1, 2),
        "weight": 0.03,
    },
    "shopping": {
        "apps": [
            ("Safari", ["Amazon", "Taobao", "JD.com"]),
        ],
        "kpm_range": (10, 40),
        "cpm_range": (50, 120),
        "duration_range": (60, 600),
        "importance_range": (1, 2),
        "weight": 0.02,
    },
    "idle": {
        "apps": [
            ("Finder", ["Desktop"]),
            ("System Preferences", ["System Preferences"]),
        ],
        "kpm_range": (0, 5),
        "cpm_range": (0, 5),
        "duration_range": (60, 300),
        "importance_range": (1, 1),
        "weight": 0.01,
    },
}


class SyntheticActivityDataset:
    """Programmatically generated activity dataset."""

    ACTIVITY_TYPES = list(ACTIVITY_PROFILES.keys())

    def __init__(self, output_dir: Path, seed: int = 42):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(seed)
        self.sessions: List[ActivitySession] = []

    def generate(
        self,
        num_sessions: int = 100,
        events_per_session: int = 100,
        generate_summaries: bool = True,
        generate_qa: bool = True,
    ) -> "SyntheticActivityDataset":
        """Generate synthetic dataset with annotations."""
        logger.info(
            f"Generating synthetic dataset: {num_sessions} sessions, "
            f"{events_per_session} events each"
        )

        self.sessions = []
        weights = [ACTIVITY_PROFILES[t]["weight"] for t in self.ACTIVITY_TYPES]

        for i in range(num_sessions):
            events = self._generate_session_events(events_per_session, weights)

            session = ActivitySession(
                session_id=f"session_{i:04d}",
                events=events,
                session_summary=self._generate_session_summary(events) if generate_summaries else "",
                qa_pairs=self._generate_qa_pairs(events) if generate_qa else [],
                importance_ranking=self._rank_by_importance(events),
            )
            self.sessions.append(session)

            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1}/{num_sessions} sessions")

        self._save()
        return self

    def _generate_session_events(
        self, n: int, weights: List[float]
    ) -> List[ActivityEvent]:
        """Generate a realistic sequence of activity events."""
        events = []
        current_time = time.time()

        # Generate sequence with some temporal coherence
        current_activity = self.rng.choices(self.ACTIVITY_TYPES, weights=weights)[0]
        activity_duration = 0
        max_activity_duration = self.rng.randint(3, 8)  # events before switching

        for _ in range(n):
            # Maybe switch activity
            activity_duration += 1
            if activity_duration >= max_activity_duration:
                # 70% chance to switch to a different activity
                if self.rng.random() < 0.7:
                    current_activity = self.rng.choices(self.ACTIVITY_TYPES, weights=weights)[0]
                activity_duration = 0
                max_activity_duration = self.rng.randint(3, 8)

            profile = ACTIVITY_PROFILES[current_activity]

            # Select app and title
            app, titles = self.rng.choice(profile["apps"])
            title = self.rng.choice(titles)

            # Generate realistic values
            duration = self.rng.uniform(*profile["duration_range"])
            kpm = self.rng.uniform(*profile["kpm_range"])
            cpm = self.rng.uniform(*profile["cpm_range"])
            importance = self.rng.randint(*profile["importance_range"])

            event = ActivityEvent(
                timestamp=current_time,
                app=app,
                title=title,
                duration_seconds=duration,
                kpm=kpm,
                cpm=cpm,
                activity_type=current_activity,
                importance_score=importance,
                reference_summary=self._generate_event_summary(app, title, current_activity),
            )
            events.append(event)

            # Advance time
            current_time += duration

        return events

    def _generate_event_summary(self, app: str, title: str, activity_type: str) -> str:
        """Generate a reference summary for a single event."""
        templates = {
            "coding": [
                f"Writing code in {app}, working on {title}",
                f"Developing in {app}: {title}",
                f"Programming session in {app}",
            ],
            "web_research": [
                f"Researching on {title} using {app}",
                f"Browsing {title} for information",
                f"Reading content on {title}",
            ],
            "document_writing": [
                f"Writing document: {title} in {app}",
                f"Editing {title}",
                f"Working on documentation in {app}",
            ],
            "email_communication": [
                f"Managing emails in {app}",
                f"Communicating via email: {title}",
                f"Email correspondence in {app}",
            ],
            "video_watching": [
                f"Watching video: {title}",
                f"Video content on {app}",
                f"Viewing media in {app}",
            ],
            "social_media": [
                f"Browsing {title}",
                f"Social media activity on {app}",
                f"Checking {title}",
            ],
            "file_management": [
                f"Organizing files in {title}",
                f"File management in Finder",
                f"Managing files and folders",
            ],
            "meeting": [
                f"In meeting via {app}",
                f"Video call: {title}",
                f"Attending meeting on {app}",
            ],
            "gaming": [
                f"Playing game on {app}",
                f"Gaming session",
                f"Entertainment: gaming",
            ],
            "shopping": [
                f"Online shopping on {title}",
                f"Browsing products on {app}",
                f"Shopping activity",
            ],
            "idle": [
                f"System idle",
                f"No active work",
                f"Away from keyboard",
            ],
        }
        return self.rng.choice(templates.get(activity_type, [f"Activity in {app}"]))

    def _generate_session_summary(self, events: List[ActivityEvent]) -> str:
        """Generate a reference summary for the entire session."""
        # Count activities
        activity_counts = {}
        for e in events:
            activity_counts[e.activity_type] = activity_counts.get(e.activity_type, 0) + 1

        # Sort by count
        sorted_activities = sorted(activity_counts.items(), key=lambda x: -x[1])
        top_activities = sorted_activities[:3]

        # Generate summary
        parts = []
        for activity, count in top_activities:
            percentage = count / len(events) * 100
            parts.append(f"{activity} ({percentage:.0f}%)")

        return f"Session focused on: {', '.join(parts)}. Total {len(events)} activities recorded."

    def _generate_qa_pairs(self, events: List[ActivityEvent]) -> List[Tuple[str, str]]:
        """Generate QA pairs for the session."""
        qa_pairs = []

        # Q1: Main activity
        activity_counts = {}
        for e in events:
            activity_counts[e.activity_type] = activity_counts.get(e.activity_type, 0) + 1
        main_activity = max(activity_counts.items(), key=lambda x: x[1])[0]
        qa_pairs.append((
            "What was the main activity during this session?",
            main_activity.replace("_", " ").title()
        ))

        # Q2: Time spent on specific activity
        if "coding" in activity_counts:
            coding_events = [e for e in events if e.activity_type == "coding"]
            total_coding_time = sum(e.duration_seconds for e in coding_events)
            qa_pairs.append((
                "How much time was spent on coding?",
                f"{total_coding_time / 60:.0f} minutes"
            ))

        # Q3: Apps used
        apps_used = list(set(e.app for e in events))
        qa_pairs.append((
            "What applications were used in this session?",
            ", ".join(apps_used[:5])
        ))

        # Q4: Most important activity
        most_important = max(events, key=lambda e: e.importance_score)
        qa_pairs.append((
            "What was the most important activity?",
            most_important.reference_summary or f"{most_important.app}: {most_important.title}"
        ))

        # Q5: Activity count
        qa_pairs.append((
            "How many distinct activities were recorded?",
            str(len(events))
        ))

        return qa_pairs

    def _rank_by_importance(self, events: List[ActivityEvent]) -> List[int]:
        """Rank events by importance score."""
        indexed = list(enumerate(events))
        sorted_indexed = sorted(indexed, key=lambda x: -x[1].importance_score)
        return [idx for idx, _ in sorted_indexed]

    def _save(self):
        """Save dataset to disk."""
        # Save sessions as JSONL
        sessions_file = self.output_dir / "sessions.jsonl"
        with open(sessions_file, "w", encoding="utf-8") as f:
            for session in self.sessions:
                f.write(json.dumps(session.to_dict(), ensure_ascii=False) + "\n")

        # Save metadata
        metadata = {
            "num_sessions": len(self.sessions),
            "total_events": sum(len(s.events) for s in self.sessions),
            "activity_types": self.ACTIVITY_TYPES,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Saved dataset: {len(self.sessions)} sessions, "
            f"{metadata['total_events']} events to {self.output_dir}"
        )

    def load(self) -> "SyntheticActivityDataset":
        """Load dataset from disk."""
        sessions_file = self.output_dir / "sessions.jsonl"
        self.sessions = []

        with open(sessions_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.sessions.append(ActivitySession.from_dict(data))

        logger.info(f"Loaded {len(self.sessions)} sessions from {self.output_dir}")
        return self


class BenchmarkDataLoader:
    """Unified data loader for benchmark evaluation."""

    def __init__(self, dataset_path: Path):
        self.dataset_path = Path(dataset_path)
        self.sessions: List[ActivitySession] = []
        self._load_sessions()

    def _load_sessions(self):
        """Load sessions from dataset directory."""
        sessions_file = self.dataset_path / "sessions.jsonl"

        if not sessions_file.exists():
            raise FileNotFoundError(f"Dataset not found: {sessions_file}")

        with open(sessions_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.sessions.append(ActivitySession.from_dict(data))

        logger.info(f"Loaded {len(self.sessions)} sessions from {self.dataset_path}")

    def get_all_events(self) -> List[ActivityEvent]:
        """Get all events across all sessions."""
        events = []
        for session in self.sessions:
            events.extend(session.events)
        return events

    def get_classification_data(self) -> Tuple[List[Dict], List[str]]:
        """Get (features, labels) for classification task."""
        features = []
        labels = []

        for event in self.get_all_events():
            features.append({
                "app": event.app,
                "title": event.title,
                "duration": event.duration_seconds,
                "kpm": event.kpm,
                "cpm": event.cpm,
            })
            labels.append(event.activity_type)

        return features, labels

    def get_summarization_data(self) -> List[Dict]:
        """Get input-reference pairs for summarization task."""
        data = []

        for session in self.sessions:
            # Event-level summaries
            for event in session.events:
                if event.reference_summary:
                    data.append({
                        "input": {
                            "app": event.app,
                            "title": event.title,
                            "activity_type": event.activity_type,
                        },
                        "reference": event.reference_summary,
                        "level": "event",
                    })

            # Session-level summary
            if session.session_summary:
                data.append({
                    "input": {
                        "events": [asdict(e) for e in session.events],
                    },
                    "reference": session.session_summary,
                    "level": "session",
                })

        return data

    def get_qa_data(self) -> List[Dict]:
        """Get QA evaluation data."""
        data = []

        for session in self.sessions:
            for question, answer in session.qa_pairs:
                data.append({
                    "session_id": session.session_id,
                    "context": [asdict(e) for e in session.events],
                    "question": question,
                    "answer": answer,
                })

        return data

    def get_ranking_data(self) -> List[Dict]:
        """Get importance ranking data."""
        data = []

        for session in self.sessions:
            if session.importance_ranking:
                data.append({
                    "session_id": session.session_id,
                    "events": [asdict(e) for e in session.events],
                    "ground_truth_ranking": session.importance_ranking,
                })

        return data


# CLI for dataset generation
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic activity dataset")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--num-sessions", type=int, default=100, help="Number of sessions")
    parser.add_argument("--events-per-session", type=int, default=100, help="Events per session")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    dataset = SyntheticActivityDataset(Path(args.output), seed=args.seed)
    dataset.generate(
        num_sessions=args.num_sessions,
        events_per_session=args.events_per_session,
    )

    print(f"Dataset generated at {args.output}")
