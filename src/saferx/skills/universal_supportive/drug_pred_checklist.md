Predict supportive ATC L4 classes only when the patient record contains evidence that directly supports the specific class.

## How to apply
- **Evidence bar** (tag after each class):
  - `[context]` a patient-specific clinical signal relevant to the class is enough (e.g., diuretic use → potassium; mechanical ventilation with coagulopathy → stress-ulcer prophylaxis).
  - `[specific]` one of the listed triggers must be documented: the drug or order is named, the lab abnormality is documented, or the specific indication/procedure is stated.
- An ongoing prior-visit medication meets either bar unless a block applies.
- Never sufficient on its own: the care setting (ICU, hospitalization, NPO, IV access, intubation), case complexity or number of diagnoses, remote history.
- Any listed **block** → do not predict.
- Within a group the first class is the default; a sibling needs its stated condition. One signal → one class (e.g., "bowel regimen" alone → A06AD, not A06AD + A06AA).
- Confidence: ≥0.70 explicit evidence; 0.50–0.69 contextual evidence; below 0.50 omit. `[specific]` classes need ≥0.65.

## Thromboprophylaxis
- **B01AB** heparin/LMWH `[context]` — heparin/enoxaparin named; VTE prophylaxis documented; critically ill, bed-bound or post-op. **Block:** active bleeding, HIT, documented anticoagulation contraindication.

## Analgesia and sedation
- **N02BE** acetaminophen `[specific]` — named; any documented pain; fever; post-op/post-procedure; multimodal analgesia; palliative/comfort care.
- **N02AA** opioids `[specific]` — major surgery (orthopedic, cardiac, abdominal, thoracic); severe trauma or burns; opioid named; MV with a documented analgesia-sedation protocol (CPOT, NRS, opioid infusion). **Block:** mild pain or minor procedure only.
- **N02BA** aspirin `[specific]` — antiplatelet use only: named as home/current med; CAD/stent/ACS/MI; stroke/TIA prevention; PAD on antiplatelet. Pain alone → N02BE. **Block:** active bleeding without documented aspirin need.
- **N01AX** propofol/ketamine `[specific]` — active MV needing sedation; intubation/RSI with agent named; tracheostomy with ongoing sedation; procedural sedation (bronchoscopy, cardioversion, endoscopy). **Block:** spontaneously breathing, CPAP/BiPAP without intubation.
- **N05CH** melatonin agonists `[specific]` — ICU sleep disruption/insomnia; delirium-prevention sleep bundle; age ≥65 with ICU agitation or circadian disruption; melatonin/ramelteon named. **Block:** non-ICU context.

## GI prophylaxis
Acid suppression:
- **A02BC** PPI `[specific]` — named as home/current med; GI bleed, peptic ulcer or GERD history; H. pylori treatment; or MV **and** (coagulopathy, shock, vasopressors or chronic liver disease) together. **Block:** ICU/MV alone.
- **A02BA** H2 blocker `[specific]` — named; PPI allergy/intolerance or contraindication with acid-suppression need; GERD with documented H2 preference; histamine-mediated allergic reaction (adjunct).
- **A02BX** sucralfate `[specific]` — named; mucosal protection when PPI/H2 contraindicated.

Antiemetics and motility:
- **A04AA** 5-HT3 antagonists `[context]` — nausea/vomiting; post-op, chemotherapy or opioid-induced nausea risk; tube feeding with nausea risk; antiemetic named.
- **A04AD** scopolamine `[specific]` — named; motion-sickness prophylaxis; PONV refractory to 5-HT3 with scopolamine ordered.
- **A03FA** metoclopramide `[specific]` — named; gastroparesis/dysmotility; post-op ileus with prokinetic ordered; tube-feeding intolerance. Nausea alone → A04AA.

Bowel regimen:
- **A06AD** osmotic laxatives `[specific]` — any opioid use this admission; constipation or bowel regimen documented; hepatic encephalopathy (lactulose); post-op immobility.
- **A06AB** senna/bisacodyl `[specific]` — named; opioid use **with** documented bowel regimen or constipation; post-op bowel protocol; immobility with constipation; bowel prep.
- **A06AA** stool softeners `[specific]` — docusate named or explicitly part of the bowel regimen.
- **A06AH** peripheral opioid antagonists `[specific]` — named; opioid-induced constipation after documented failure of conventional laxatives.
- **Block (all laxatives):** diarrhea, ileus, bowel obstruction.

## Electrolyte, fluid and nutrition support
- **A12BA** potassium `[context]` — named; loop/thiazide diuretic use; arrhythmia with electrolyte monitoring; renal impairment with electrolyte management. **Block:** hyperkalemia.
- **A12AA** calcium `[context]` — named; CKD/renal impairment; diuretic use; massive transfusion/citrate; electrolyte replacement in an ICU patient.
- **A12CC** magnesium `[specific]` — hypomagnesemia; Mg repletion ordered; torsades/QT prolongation; arrhythmia with Mg monitoring; alcohol withdrawal with Mg replacement; diuretic-induced Mg loss.
- **V06DC** dextrose `[context]` — enteral/tube feeding; parenteral nutrition; insulin infusion (DKA, hyperkalemia protocol); hypoglycemia; IV dextrose named.
- **B05XA** electrolyte solutions `[specific]` — named crystalloid (NS, LR, PlasmaLyte); active fluid resuscitation; IV electrolyte correction plan; severe electrolyte imbalance.
- **B05AA** albumin `[specific]` — infusion named/ordered; SBP with albumin protocol; large-volume paracentesis; hepatorenal syndrome. **Block:** low albumin lab alone.
- **A11DA** thiamine `[specific]` — alcohol history with thiamine prophylaxis; Wernicke risk; refeeding syndrome.
- **A11CC** vitamin D `[specific]` — deficiency; supplementation order; CKD-MBD.
- **B03BA** vitamin B12 `[specific]` — named; B12 deficiency or pernicious anemia; post-gastrectomy/bariatric supplementation.

## Other supportive care
- **D04AB** topical anesthetics `[specific]` — intubation or throat procedure with topical anesthetic mentioned.
- **R02AA** throat antiseptics `[specific]` — chlorhexidine named or oral-care protocol documented.
- **R02AD** throat local anesthetics `[specific]` — throat spray/lozenge named; post-intubation/extubation sore throat or endoscopy prep with topical anesthetic ordered.
- **R02AX** other throat preparations `[specific]` — lozenge/spray/gargle named; post-intubation sore throat with treatment ordered; pharyngitis/laryngitis with topical treatment.
- **J07BB** influenza vaccine `[specific]` — vaccine ordered; or adult admitted Oct–Apr without contraindication. **Block:** severe egg allergy, Guillain-Barré within 6 weeks, pediatric patient without vaccine mention.
- **J01MA** fluoroquinolones `[specific]` — complicated UTI/pyelonephritis; atypical or community-acquired pneumonia needing levofloxacin; prostatitis; osteomyelitis/bone-joint infection; named. **Block:** infection without specific site.
