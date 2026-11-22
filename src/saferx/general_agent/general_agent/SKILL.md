# General Agent Skill

This skill handles ALL medication domains for a patient admission as a single generalist agent. It covers every therapeutic area — cardiovascular, respiratory, infectious disease, endocrine/metabolic, gastrointestinal, musculoskeletal, oncology/hematology, neurological, obstetrics, and universal supportive care.

Core lens:
- Disease-specific treatments: antihypertensives, antiarrhythmics, bronchodilators, antibiotics, antidiabetics, chemotherapy support, anticonvulsants, etc.
- Supportive care: VTE prophylaxis, electrolyte repletion, GI prophylaxis, pain/sedation, nutritional support
- Prior medication continuation when clinically appropriate
- Procedure-related medications: peri-operative antibiotics, anticoagulation, analgesia

The agent must consider the full clinical picture and predict all ATC L4 medication classes that are plausibly present during the current admission, across all domains.
