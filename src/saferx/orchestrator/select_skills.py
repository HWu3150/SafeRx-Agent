from __future__ import annotations

from typing import Any

from ..router import RoutingConfig, load_routing_config


def select_skills(
    routing_result: dict[str, Any],
    routing_config: RoutingConfig | None = None,
    skill_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = routing_config or load_routing_config()
    ranked = list(routing_result.get("ranked_domains", []))
    if not ranked:
        return []

    allowed_ids = set(skill_registry.keys()) if skill_registry else None
    primary_id = routing_result.get("primary_domain_id")
    primary_score = float(routing_result.get("primary_score", 0.0) or 0.0)
    is_mixed = bool(routing_result.get("is_mixed"))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for entry in ranked[: max(config.activate_top_n, 1)]:
        domain_id = entry.get("id")
        if not domain_id:
            continue
        if allowed_ids is not None and domain_id not in allowed_ids:
            continue

        score = float(entry.get("score", 0.0) or 0.0)
        should_activate = False
        if domain_id == primary_id:
            should_activate = True
        elif score >= config.min_skill_score:
            score_gap = primary_score - score
            should_activate = (
                is_mixed
                or score >= config.secondary_score_floor
                or score_gap <= config.selection_margin
            )

        if not should_activate or domain_id in selected_ids:
            continue

        selected.append(
            {
                "skill_id": domain_id,
                "skill_name": entry.get("name", domain_id),
                "routing_score": score,
                "rank": int(entry.get("rank", len(selected) + 1)),
                "activation_reason": "primary"
                if domain_id == primary_id
                else "mixed_or_high_secondary_score",
            }
        )
        selected_ids.add(domain_id)

        if len(selected) >= config.max_active_skills:
            break

    return selected
