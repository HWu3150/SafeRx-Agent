from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExpertiseSkill:
    skill_id: str
    name: str
    root_dir: Path
    skill_md: str
    summarize_template: str
    drug_pred_checklist: str
    critic_focus: str
    report_template: str


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _skill_name(skill_md: str, fallback: str) -> str:
    """Display name from the SKILL.md heading, e.g. '# Cardiovascular Skill' -> 'Cardiovascular'."""
    first = skill_md.splitlines()[0].lstrip("#").strip() if skill_md else ""
    return first.removesuffix(" Skill").strip() or fallback


def _load_skill_from_dir(path: Path) -> ExpertiseSkill:
    skill_md = _read_text(path / "SKILL.md")
    return ExpertiseSkill(
        skill_id=path.name,
        name=_skill_name(skill_md, path.name.replace("_", " ").title()),
        root_dir=path,
        skill_md=skill_md,
        summarize_template=_read_text(path / "summarize_template.md"),
        drug_pred_checklist=_read_text(path / "drug_pred_checklist.md"),
        critic_focus=_read_text(path / "critic_focus.md"),
        report_template=_read_text(path / "report_template.md"),
    )


def discover_skills(root_dir: str | Path | None = None) -> dict[str, ExpertiseSkill]:
    base_dir = Path(root_dir) if root_dir else SKILLS_DIR
    skills: dict[str, ExpertiseSkill] = {}
    for child in sorted(base_dir.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "SKILL.md").exists():
            continue
        skills[child.name] = _load_skill_from_dir(child)
    return skills


def load_skill(skill_id: str, root_dir: str | Path | None = None) -> ExpertiseSkill:
    skills = discover_skills(root_dir=root_dir)
    if skill_id not in skills:
        raise KeyError(f"Unknown expertise skill: {skill_id}")
    return skills[skill_id]
