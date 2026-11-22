Predict ATC-L4 medication classes for the current inpatient admission as a single generalist agent. Use the patient summary, prior-visit medication evidence, diagnosis-linked medication evidence, and the closed ATC-L4 vocabulary. The prediction should cover both disease-directed therapy and common inpatient supportive care, while requiring patient-specific evidence for each medication class.

The agent should reason over the case by identifying active problems and chronic conditions, mapping each to plausible medication needs, checking whether prior medications should be continued, and considering medications associated with procedures, critical illness, infection management, pain control, thrombosis prevention, gastrointestinal protection, electrolyte correction, nutrition, and bowel care. Medication classes should be included only when supported by the current visit context, longitudinal history, or explicit evidence in the record.

The agent should prefer adequate coverage of clinically supported medication needs, but should avoid adding classes solely because they are common in inpatient care. Confidence should reflect the strength of evidence: higher confidence is reserved for explicitly documented medications or indications, while lower confidence is used for plausible but indirect clinical support.

## How to apply
- Work problem by problem: for each active problem or chronic condition, take the default class in its line below; then check prior-visit medications for continuation; then supportive care.
- Predict a class when its trigger is documented for the current admission, or it is an ongoing prior-visit medication. Not enough: a diagnosis name without admission-level context, remote history, or the care setting alone.
- Any **block** → do not predict.
- In each line the first class is the default; a class marked *second-line* needs its stated condition. One indication → one class unless a combination is noted.
- Confidence: ≥0.70 explicit medication or indication; 0.50–0.69 indirect support; below 0.50 omit.

## Disease-directed therapy
- **Cardiovascular**
  - AF/flutter or rate control → C07AB; *second-line* C01BD for rhythm control.
  - HF or HTN → C09AA; *second-line* C09CA for ACE-inhibitor intolerance; C08CA for HTN without HF.
  - Volume overload → C03CA; *second-line* C03DA for HFrEF or cirrhotic ascites.
  - CAD, stent, ACS or ischemic stroke → B01AC + C10AA.
  - Therapeutic anticoagulation (AF, VTE, mechanical valve) → B01AF; *second-line* B01AA for mechanical valve or warfarin named.
  - Shock on vasopressors → C01CA.
  - **Block:** C07AB in cardiogenic shock or symptomatic bradycardia; antithrombotics in active bleeding.
- **Respiratory**
  - Asthma/COPD exacerbation → R03AC + H02AB; R03BA when maintained on an inhaled steroid.
- **Infection** (documented or treated infection only)
  - Pneumonia → J01DD + J01FA.
  - Sepsis, hospital-acquired or unknown source → J01CR + J01XA; *second-line* J01DH for ESBL or recent broad-spectrum failure.
  - Intra-abdominal or C. difficile → P01AB.
  - UTI → J01DD; *second-line* J01MA.
  - Fungal infection → J02AC. HSV/VZV/CMV → J05AB.
- **Endocrine/metabolic**
  - Type 2 diabetes on oral therapy → A10BA; *second-line* A10BB when named.
  - Hypothyroidism → H03AA.
  - Adrenal insufficiency or chronic steroid use → H02AB.
- **GI/hepatic**
  - GI bleed, peptic ulcer or GERD → A02BC.
  - Hepatic encephalopathy → A06AD + A07AA.
  - Varices or portal hypertension → C07AA.
- **Neuro/psychiatric**
  - Seizure or epilepsy → N03AX.
  - Alcohol withdrawal → N05BA.
  - Delirium or agitation → N05AD; *second-line* N05AH for a prior or named atypical antipsychotic.
  - Depression on treatment → N06AB.
- **Oncology/hematology**
  - Chemotherapy → A04AA. Neutropenia → L03AA.
  - Anemia of CKD → B03XA. B12 deficiency → B03BA.
- **Obstetrics**
  - Delivery or postpartum hemorrhage → H01BB.
  - Pre-eclampsia → A12CC.
- **Pain**
  - Any documented pain or fever → N02BE.
  - Major surgery, trauma or severe pain → N02AA.
  - *Second-line* M01AE for musculoskeletal pain. **Block:** AKI, CKD, GI bleed or anticoagulation.

## Supportive care
- **VTE prophylaxis** → B01AB: critically ill, bed-bound or post-op. **Block:** therapeutic anticoagulation, active bleeding, HIT.
- **Stress-ulcer prophylaxis** → A02BC: mechanical ventilation **and** (coagulopathy, shock or chronic liver disease). ICU alone is not enough.
- **Bowel care** → A06AD: opioid use or documented constipation. **Block:** diarrhea, ileus, obstruction.
- **Nausea** → A04AA: nausea, post-op or opioid-related.
- **Electrolytes**
  - Hypokalemia or loop-diuretic use → A12BA.
  - Hypomagnesemia → A12CC.
  - Hypocalcemia or massive transfusion → A12AA.
- **Fluids and nutrition**
  - IV fluid resuscitation or electrolyte infusions → B05XA.
  - Enteral or parenteral nutrition, or hypoglycemia → V06DC.
