from __future__ import annotations

import re
from typing import Sequence


SKILL_ALIASES: dict[str, tuple[str, ...]] = {
    # Languages
    "python": ("python",),
    "java": ("java",),
    "javascript": ("javascript", "js"),
    "html": ("html",),
    "css": ("css",),
    "sql": ("sql",),
    
    # Web Frameworks & Libraries
    "react": ("react", "reactjs"),
    "angular": ("angular",),
    "node.js": ("node.js", "nodejs", "node"),
    "express": ("express",),
    "django": ("django",),
    "flask": ("flask",),
    "spring": ("spring", "spring framework", "spring boot"),
    "jquery": ("jquery",),
    "bootstrap": ("bootstrap",),
    
    # Frontend
    "html": ("html",),
    "css": ("css",),
    "frontend": ("frontend", "front end"),
    "tailwind": ("tailwind",),
    "material design": ("material design", "material ui"),
    
    # Backend & Architecture
    "backend": ("backend", "back end"),
    "rest apis": ("rest api", "rest apis", "restful"),
    "microservices": ("microservices", "microservice"),
    
    # Databases
    "sql": ("sql",),
    "mysql": ("mysql",),
    "postgresql": ("postgresql", "postgres"),
    "oracle": ("oracle",),
    "mongodb": ("mongodb", "mongo"),
    
    # ORM & Persistence
    "jpa": ("jpa",),
    "hibernate": ("hibernate",),
    
    # ML/AI
    "machine learning": ("ml", "machine learning"),
    "nlp": ("nlp", "natural language processing"),
    "tensorflow": ("tensorflow",),
    "pytorch": ("pytorch",),
    "keras": ("keras",),
    "pandas": ("pandas",),
    "numpy": ("numpy",),
    "scikit-learn": ("scikit-learn", "sklearn"),
    
    # DevOps & Tools
    "docker": ("docker",),
    "git": ("git",),
    "linux": ("linux",),
    "aws": ("aws", "amazon web services"),
    "maven": ("maven",),
    "gradle": ("gradle",),
    "jenkins": ("jenkins",),
    "ci/cd": ("ci/cd", "cicd", "continuous integration"),
    
    # Testing
    "junit": ("junit",),
    "mockito": ("mockito",),
    "testing": ("testing", "test", "unit test"),
    
    # Data & Analytics
    "excel": ("excel",),
    "tableau": ("tableau",),
    "spark": ("spark",),
    "hadoop": ("hadoop",),
    
    # Methodologies
    "agile": ("agile",),
    "scrum": ("scrum",),
    "oop": ("oop", "object oriented", "object-oriented programming"),
}

ROLE_SKILL_HINTS: dict[str, tuple[str, ...]] = {
    "web developer": ("html", "css", "javascript", "react", "frontend"),
    "frontend developer": ("html", "css", "javascript", "react", "frontend"),
    "backend developer": ("python", "sql", "django", "flask", "backend"),
    "full stack developer": ("html", "css", "javascript", "react", "node.js", "frontend", "backend"),
}


def _match_skill(text: str, skill_aliases: Sequence[str]) -> bool:
    for alias in skill_aliases:
        pattern = re.escape(alias.lower()).replace(r"\ ", r"[\s\-]+")
        if re.search(rf"\b{pattern}\b", text):
            return True
    return False


def extract_skills(text: str) -> tuple[list[str], list[str]]:
    """Extract predefined skills from text using simple keyword and role matching."""
    normalized_text = re.sub(r"\s+", " ", (text or "").lower())

    matched_skills: list[str] = []
    for skill, aliases in SKILL_ALIASES.items():
        if _match_skill(normalized_text, aliases):
            matched_skills.append(skill)

    for role, role_skills in ROLE_SKILL_HINTS.items():
        if _match_skill(normalized_text, (role,)):
            for skill in role_skills:
                if skill not in matched_skills:
                    matched_skills.append(skill)

    missing_skills = [skill for skill in SKILL_ALIASES if skill not in matched_skills]
    return matched_skills, missing_skills