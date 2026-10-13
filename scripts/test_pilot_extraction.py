"""
Test script to verify pilot subcohort event extraction on 2 sample patients.
"""

from pathlib import Path
from src.data.extract_events import CohortEventExtractor


def main():
    manifest_path = Path("data/pilot_cohort_manifest.csv")
    extractor = CohortEventExtractor(manifest_path)
    
    # Pick first 2 subjects from manifest
    target_sids = list(extractor.patients.keys())[:2]
    print(f"Testing extraction on subject_ids: {target_sids}")
    
    # Extract admissions & demographics
    print("Extracting admissions...")
    demog_and_adm = extractor.extract_admissions_and_demographics(set(target_sids))
    
    # Extract ECGs
    print("Extracting ECG events...")
    ecg_events = extractor.extract_ecg_events(set(target_sids))
    
    # Extract CXRs
    print("Extracting CXR events...")
    cxr_events = extractor.extract_cxr_events(set(target_sids))
    
    # Extract Labs
    print("Extracting Lab events...")
    lab_events = extractor.extract_lab_events(set(target_sids))
    
    for sid in target_sids:
        prefix, adm_evs = demog_and_adm[sid]
        all_evs = adm_evs + ecg_events.get(sid, []) + cxr_events.get(sid, []) + lab_events.get(sid, [])
        journey = extractor.assemble_patient_journey(sid, prefix, all_evs)
        
        print("\n" + "=" * 60)
        print(f"PATIENT JOURNEY: {sid}")
        print(f"Demographics: {journey.demographics}")
        print(f"Total events extracted: {journey.total_events}")
        print(f"Eligible (>=64 events): {journey.is_eligible()}")
        
        print("First 5 chronological events:")
        for ev in journey.events[:5]:
            print(f"  [ord={ev.ordinal:3d}] t={ev.charttime:10.2f}h (age={ev.age:.2f}y) | "
                  f"cat={ev.category_id.name:20s} | mod={ev.modality_id.name:10s} | "
                  f"d_prev={ev.delta_prev:6.2f}h | d_next={ev.delta_next:6.2f}h | "
                  f"has_val={ev.has_value}")
            
        print("Last 3 events:")
        for ev in journey.events[-3:]:
            print(f"  [ord={ev.ordinal:3d}] t={ev.charttime:10.2f}h (age={ev.age:.2f}y) | "
                  f"cat={ev.category_id.name:20s} | mod={ev.modality_id.name:10s} | "
                  f"d_prev={ev.delta_prev:6.2f}h | d_next={ev.delta_next:6.2f}h | "
                  f"has_val={ev.has_value}")


if __name__ == "__main__":
    main()
