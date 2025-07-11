#!/usr/bin/env python3

# Mock file for medcalc_rules with traces using 1.5B model (GPU memory friendly)
# This file provides the detailed execution traces for the medcalc_rules function

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import mock

# Input examples with complete medical case data
INPUT1 = """We present an 85-year-old male with a past medical history of obesity, type two diabetes, atrial fibrillation, diastolic heart failure, chronic kidney disease (CKD) stage three, and coronary artery disease. The patient had a history of non-ST elevation myocardial infarction (NSTEMI) four weeks before the current admission, requiring a drug-eluting stent (DES) to the left anterior descending (LAD) artery. He also had a history of significant gastrointestinal bleed in the past month, for which apixaban was stopped. The patient was admitted to our hospital with worsening shortness of breath and found to have bilateral pleural effusions, right greater than left. He was afebrile and did not have any symptoms of pneumonia. The patient was started on IV furosemide and had an initial diagnostic, small-bore, ultrasound-guided tap from the right pleural effusion that was uneventful and yielded straw-colored 1000 mL of fluid. The pleural fluid analysis was mildly exudative based on Light's lactate dehydrogenase (LDH) criteria, but cytology was negative as well as Gram stain, bacterial, and fungal cultures. Autoimmune screening, including anti-nuclear antibody (ANA) and extractable nuclear antigen (ENA), was negative. In anticipation of a potential repeat pleural tap, the patient's aspirin was stopped. One week later, the patient was getting more short of breath, and chest X-ray revealed recurrent bilateral effusions worse on the right side. Echocardiogram showed features of diastolic dysfunction, with a left ventricular ejection fraction of 55% and no significant valvular disease. Arterial blood gas (ABG) was suggestive of hypercapnic respiratory failure; thus, he was started on non-invasive ventilation (NIV) and shifted to the ICU. He underwent a second, uneventful pleural tap on the right side, that yielded 1500 mL of straw-colored fluid. Analysis again showed an exudate with negative bacterial, fungal cultures, and cytology. CT scan of the chest showed basal atelectasis with significant pleural effusions, no lung masses, or lymph nodes enlargement (Figure ). The patient was transferred to the step-down unit, completed a 10-day course of antibiotics for possible community-acquired pneumonia, although sputum and blood cultures remained negative. One week later, the patient again clinically deteriorated and was admitted to ICU with hypercapnic respiratory failure and worsening pleural effusions. He initially required continuous bilevel positive airway pressure (BiPAP) ventilation until he stabilized. He had a third, right-sided thoracocentesis under ultrasound-guidance from a posterior approach, atraumatic, and yielded 1500 mL of clear thin yellow fluid. The patient had a follow-up chest X-ray 20 min later that showed improvement in the previously seen right-sided pleural effusion and no pneumothorax. However, two hours later, the patient was suddenly getting sweaty, tachypneic, lethargic while on BiPAP. His blood pressure (BP) was reading 60/40 mmHg, heart rate was dropping to 40 bpm, and he was less responsive. There was minimal air entry on auscultating the right chest with dullness to percussion, and no signs of any pain or tenderness. ABG showed hypoxia and a stable PCO2 level. Immediate chest X-ray showed a new 3.5 cm hemothorax on the right side (Figure ).
He was given 1 L of IV fluids, norepinephrine for BP support, and an urgent right-sided IC surgical drain was inserted, which immediately drained 1.7 L of blood.
The patient had a central line insertion, and three units of packed red blood cells (PRBC) transfused. Repeat hemoglobin after the second transfused unit was still below baseline at 6 g/L (from 8 g/L). He clinically started to improve after the hemothorax drainage, was more responsive, and pressor requirement was decreased. Overnight he received two units of platelets, and clopidogrel was stopped. The next day, the IC surgical drain still had around 800 mL output of serosanguinous fluid, but he was clinically back to his baseline as before this event. CT scan of the chest with IV contrast did not show any active bleeding into the pleural space but showed complex right-sided effusion. The IR team performed IC arteries angiogram to diagnose the site of the injured vessel. That was done within 24 h of the event and was negative. The cardiothoracic surgery team was consulted for possible video-assisted thoracoscopic surgery (VATS) procedure in the right pleural space. The patient continued to improve, and after a few days we removed the surgical drain, and transferred him out to the step-down unit with cardiothoracic surgical team following up on his care.

What is the patient's CHA2DS2-VASc Score?

Calculate CHA2DS2-VASc Score for Atrial Fibrillation Stroke Risk by summing the results of the following rules:
    Age: < 65 years = 0 points, 65-74 years = +1 point, ≥ 75 years = +2 points
    Sex: Female = +1 point, Male = 0 points
    Congestive Heart Failure (CHF) history: No = 0 points, Yes = +1 point
    Hypertension history: No = 0 points, Yes = +1 point
    Stroke, Transient Ischemic Attack (TIA), or Thromboembolism history: No = 0 points, Yes = +2 points
    Vascular disease history (previous myocardial infarction, peripheral artery disease, or aortic plaque): No = 0 points, Yes = +1 point
    Diabetes history: No = 0 points, Yes = +1 point"""

QUES1 = "What is the patient's CHA2DS2-VASc Score?"
RULES1 = [
    "Age: < 65 years = 0 points, 65-74 years = +1 point, ≥ 75 years = +2 points",
    "Sex: Female = +1 point, Male = 0 points",
    "Congestive Heart Failure (CHF) history: No = 0 points, Yes = +1 point",
    "Hypertension history: No = 0 points, Yes = +1 point",
    "Stroke, Transient Ischemic Attack (TIA), or Thromboembolism history: No = 0 points, Yes = +2 points",
    "Vascular disease history (previous myocardial infarction, peripheral artery disease, or aortic plaque): No = 0 points, Yes = +1 point",
    "Diabetes history: No = 0 points, Yes = +1 point"
]
DATAPOINTS1 = [
    "85-year-old male",
    "male",
    "diastolic heart failure",
    "No hypertension mentioned",
    "No stroke/TIA mentioned", 
    "non-ST elevation myocardial infarction (NSTEMI)",
    "type two diabetes"
]

INPUT2 = """Fifty-one-year-old Caucasian female was admitted to our hospital with complaints of fever and chills for three days and low back pain for two days. Past medical history included recent spinal abscess followed by implantation of spinal nerve stimulator, mitral valve prolapse and GERD. Her medications included omeprazole and acetaminophen as needed for chronic back pain. On physical examination, patient appeared sick. Vital signs: BP 80/50 mmHg, Pulse 120/min, respiratory rate 24/min. No skin lesions; neck was supple. Jugular veins were distended up to angle of the ear with prominent 'a' and 'v' waves. Her P2 was loud; No murmurs or gallop; lung fields were clear with normal vesicular breath sounds. Abdomen was soft and non tender. She had increased warmth and tenderness at the level of D10 - D12 vertebrae. Neurological examination was normal. Her labs revealed WBC count of 15,000 with bandaemia of 20%, lactic acid of 45 mgs%. Her EKG revealed sinus tachycardia. Serial troponin levels were normal. She was treated with broad spectrum antibiotics, intravenous fluids and dopamine as infusion. Her blood culture was positive for Staphylococcus aureus. Her echocardiogram revealed small left ventricle with ejection fraction of 65%, flattened interventricular septum, normal left atrium (LA), bowing of the interatrial septum to the LA, right atrial (RA) and RV enlargement with severe hypokinesia of RV (). Estimated RV systolic pressure was 75 mmHg. She also had TEE to look for any vegetation. There was no vegetation. Her echo examination 8 months ago was normal. High resolution Chest computerized tomography (CT) revealed normal lungs and CT angiogram revealed no evidence of pulmonary embolism. The patient underwent left and right heart catheterization10 days later. Coronary arteries and LV function were normal. RA pressure 6 mmHg, RV, 36/11 mmHg, pulmonary artery 36/11 mmHg, mean 24 mmHg, indicating significant improvement of pulmonary hypertension. Echo on the next day revealed markedly improved RA and RV dilatation (). Patient had a full recovery on discharge.

What is the patient's score of Wells' criteria for Pulmonary Embolism?

Calculate Wells' Criteria for Pulmonary Embolism by summing the results of the following rules:
    Clinical signs and symptoms of DVT: No = 0 points, Yes = +3 points
    PE is #1 diagnosis OR equally likely: No = 0 points, Yes = +3 points
    Heart rate > 100: No = 0 points, Yes = +1.5 points
    Immobilization at least 3 days OR surgery in the previous 4 weeks: No = 0 points, Yes = +1.5 points
    Previous, objectively diagnosed PE or DVT: No = 0 points, Yes = +1.5 points
    Hemoptysis: No = 0 points, Yes = +1 point
    Malignancy with treatment within 6 months or palliative: No = 0 points, Yes = +1 point"""

QUES2 = "What is the patient's score of Wells' criteria for Pulmonary Embolism?"
RULES2 = [
    "Clinical signs and symptoms of DVT: No = 0 points, Yes = +3 points",
    "PE is #1 diagnosis OR equally likely: No = 0 points, Yes = +3 points",
    "Heart rate > 100: No = 0 points, Yes = +1.5 points",
    "Immobilization at least 3 days OR surgery in the previous 4 weeks: No = 0 points, Yes = +1.5 points",
    "Previous, objectively diagnosed PE or DVT: No = 0 points, Yes = +1.5 points",
    "Hemoptysis: No = 0 points, Yes = +1 point",
    "Malignancy with treatment within 6 months or palliative: No = 0 points, Yes = +1 point"
]
DATAPOINTS2 = [
    "No DVT signs mentioned",
    "CT angiogram revealed no evidence of pulmonary embolism",
    "Pulse 120/min",
    "recent spinal abscess followed by implantation of spinal nerve stimulator",
    "No previous PE/DVT mentioned",
    "No hemoptysis mentioned",
    "No malignancy mentioned"
]

INPUTS = [INPUT1, INPUT2]

analyze_input_mockdict = {
    INPUT1: (QUES1, RULES1, DATAPOINTS1),
    INPUT2: (QUES2, RULES2, DATAPOINTS2)
}

def get_data_proxy(rule, datapoints):
    if rule == RULES1[0]:  # Age rule
        return DATAPOINTS1[0]
    elif rule == RULES1[1]:  # Sex rule
        return DATAPOINTS1[1]
    elif rule == RULES1[2]:  # CHF rule
        return DATAPOINTS1[2]
    elif rule == RULES1[3]:  # Hypertension rule
        return DATAPOINTS1[3]
    elif rule == RULES1[4]:  # Stroke/TIA rule
        return DATAPOINTS1[4]
    elif rule == RULES1[5]:  # Vascular disease rule
        return DATAPOINTS1[5]
    elif rule == RULES1[6]:  # Diabetes rule
        return DATAPOINTS1[6]
    elif rule == RULES2[0]:  # DVT signs
        return DATAPOINTS2[0]
    elif rule == RULES2[1]:  # PE diagnosis
        return DATAPOINTS2[1]
    elif rule == RULES2[2]:  # Heart rate
        return DATAPOINTS2[2]
    elif rule == RULES2[3]:  # Immobilization/surgery
        return DATAPOINTS2[3]
    elif rule == RULES2[4]:  # Previous PE/DVT
        return DATAPOINTS2[4]
    elif rule == RULES2[5]:  # Hemoptysis
        return DATAPOINTS2[5]
    elif rule == RULES2[6]:  # Malignancy
        return DATAPOINTS2[6]

check_rule_mockdict = {
    # CHA2DS2-VASc scoring
    (RULES1[0], DATAPOINTS1[0]): 2,    # Age ≥75: +2 points
    (RULES1[1], DATAPOINTS1[1]): 0,    # Male: 0 points
    (RULES1[2], DATAPOINTS1[2]): 1,    # CHF: +1 point
    (RULES1[3], DATAPOINTS1[3]): 0,    # No hypertension: 0 points
    (RULES1[4], DATAPOINTS1[4]): 0,    # No stroke/TIA: 0 points
    (RULES1[5], DATAPOINTS1[5]): 1,    # MI (vascular disease): +1 point
    (RULES1[6], DATAPOINTS1[6]): 1,    # Diabetes: +1 point
    
    # Wells scoring
    (RULES2[0], DATAPOINTS2[0]): 0,    # No DVT signs: 0 points
    (RULES2[1], DATAPOINTS2[1]): 0,    # PE not #1 diagnosis: 0 points
    (RULES2[2], DATAPOINTS2[2]): 1.5,  # HR 120 > 100: +1.5 points
    (RULES2[3], DATAPOINTS2[3]): 1.5,  # Recent surgery: +1.5 points
    (RULES2[4], DATAPOINTS2[4]): 0,    # No previous PE/DVT: 0 points
    (RULES2[5], DATAPOINTS2[5]): 0,    # No hemoptysis: 0 points
    (RULES2[6], DATAPOINTS2[6]): 0     # No malignancy: 0 points
}

def accumulate_score_proxy(acc, to_add):
    return acc + to_add

# Mock decorators and functions
@mock.dictmock(analyze_input_mockdict)
def analyze_input(input_str: str) -> tuple[str, list[str], list[str]]:
    """Accepts an input and extracts the question being asked, a list of rules to follow to answer the question, and the given relevant data."""
    ...

@mock.proxymock(get_data_proxy)
def get_data(rule: str, relevant_data: list[str]) -> str:
    """Accepts a rule and a set of relevant data, and extracts the datapoint required to evaluate the rule."""
    ...

@mock.dictmock(check_rule_mockdict)
def check_rule(rule: str, relevant_data: str) -> int:
    """Accepts a rule and the data necessary to determine how the rule applies, and returns an integer value corresponding to the rule."""
    ...

@mock.proxymock(accumulate_score_proxy)
def accumulate_score(accumulator: int, to_add: int) -> int:
    """Accepts an accumulator and a number to be added to it, and returns the sum."""
    ...

@mock.nullmock(sample_inputs=INPUTS)
def medcalc_rules(input_str):
    """Follow medical rules to calculate a score based on patient data.

    ###DOCTESTS FOR medcalc_rules
    """
    print(f"TRACE: medcalc_rules called with input: {input_str[:100]}...")
    
    print("TRACE: Calling analyze_input...")
    question, rules, datapoints = analyze_input(input_str)
    print(f"TRACE: analyze_input returned question='{question}', {len(rules)} rules, {len(datapoints)} datapoints")
    
    acc = 0
    print(f"TRACE: Starting accumulation with acc={acc}")
    
    for i, rule in enumerate(rules):
        print(f"TRACE: Processing rule {i+1}/{len(rules)}: {rule}")
        
        print(f"TRACE: Calling get_data for rule...")
        relevant_data = get_data(rule, datapoints)
        print(f"TRACE: get_data returned: {relevant_data}")
        
        print(f"TRACE: Calling check_rule...")
        to_add = check_rule(rule, relevant_data)
        print(f"TRACE: check_rule returned: {to_add}")
        
        print(f"TRACE: Calling accumulate_score with acc={acc}, to_add={to_add}")
        acc = accumulate_score(acc, to_add)
        print(f"TRACE: accumulate_score returned: {acc}")
    
    print(f"TRACE: Final calculation complete")
    print(f"Final answer: {acc}")
    return acc

if __name__ == '__main__':
    mock.main(locals()) 